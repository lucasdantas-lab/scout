"""Async client for the API-Football REST API.

Handles fixture fetching, per-fixture statistics, upcoming fixtures,
and bulk ingestion with rate limiting and exponential back-off.
"""

import asyncio
import logging
from typing import Any

import httpx

from config import API_FOOTBALL_KEY, LEAGUE_ID

logger = logging.getLogger(__name__)

_BASE_URL = "https://v3.football.api-sports.io"
_RATE_LIMIT = 10          # max requests per second
_MAX_RETRIES = 5
_BACKOFF_BASE = 1.5       # seconds


class APIFootballClient:
    """Async HTTP client wrapping API-Football v3."""

    def __init__(self) -> None:
        self._headers = {
            "x-rapidapi-host": "v3.football.api-sports.io",
            "x-rapidapi-key": API_FOOTBALL_KEY,
        }
        self._semaphore = asyncio.Semaphore(_RATE_LIMIT)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any]) -> list[dict]:
        """Execute a GET request with retry/back-off logic.

        Args:
            path: API endpoint path (e.g. '/fixtures').
            params: Query parameters dict.

        Returns:
            The 'response' list from the API payload.

        Raises:
            RuntimeError: When the API returns no 'response' field or
                the field is empty after all retries are exhausted.
        """
        url = f"{_BASE_URL}{path}"
        async with self._semaphore:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    async with httpx.AsyncClient(
                        headers=self._headers, timeout=30.0
                    ) as client:
                        resp = await client.get(url, params=params)
                        resp.raise_for_status()
                        payload = resp.json()

                    if "response" not in payload:
                        raise RuntimeError(
                            f"Missing 'response' key in API payload for {url} "
                            f"(params={params}). Raw: {payload}"
                        )
                    if not payload["response"]:
                        logger.debug(
                            "Empty 'response' for %s params=%s", url, params
                        )
                        return []

                    return payload["response"]

                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    wait = _BACKOFF_BASE**attempt
                    logger.warning(
                        "Request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        exc,
                        wait,
                    )
                    if attempt == _MAX_RETRIES:
                        raise RuntimeError(
                            f"API request to {url} failed after {_MAX_RETRIES} retries."
                        ) from exc
                    await asyncio.sleep(wait)

        return []  # unreachable, satisfies type checker

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def fetch_fixtures(
        self, season: int, league_id: int = LEAGUE_ID
    ) -> list[dict]:
        """Fetch all fixtures for a given season and league.

        Args:
            season: Four-digit season year (e.g. 2023).
            league_id: API-Football league identifier.

        Returns:
            List of normalised match dicts compatible with the `matches` table.
        """
        raw = await self._get(
            "/fixtures", {"league": league_id, "season": season}
        )
        return [self._normalise_fixture(f) for f in raw]

    async def fetch_statistics(self, fixture_id: int) -> dict | None:
        """Fetch per-fixture statistics.

        Args:
            fixture_id: API-Football fixture id.

        Returns:
            Dict compatible with the `match_stats` table, or None when no
            statistics are available for this fixture.
        """
        raw = await self._get("/fixtures/statistics", {"fixture": fixture_id})
        if not raw:
            return None
        return self._normalise_statistics(fixture_id, raw)

    async def fetch_upcoming(
        self, league_id: int = LEAGUE_ID, next_n: int = 10
    ) -> list[dict]:
        """Fetch the next N scheduled (not yet played) fixtures.

        Args:
            league_id: API-Football league identifier.
            next_n: Maximum number of upcoming fixtures to retrieve.

        Returns:
            List of normalised match dicts.
        """
        raw = await self._get(
            "/fixtures", {"league": league_id, "next": next_n}
        )
        return [self._normalise_fixture(f) for f in raw]

    async def bulk_ingest(self, seasons: list[int]) -> list[dict]:
        """Orchestrate full ingestion for multiple seasons.

        For every finished fixture (status='FT') also fetches statistics.
        Applies a rate limit of ``_RATE_LIMIT`` concurrent requests.

        Args:
            seasons: List of season years to ingest.

        Returns:
            List of dicts, each containing 'match' and optionally 'stats' keys.
        """
        results: list[dict] = []

        for season in seasons:
            logger.info("Ingesting season %d …", season)
            fixtures = await self.fetch_fixtures(season)
            logger.info(
                "Season %d: %d fixtures found.", season, len(fixtures)
            )

            finished = [f for f in fixtures if f.get("status") == "FT"]
            logger.info(
                "Season %d: %d finished fixtures.", season, len(finished)
            )

            stats_tasks = [
                self.fetch_statistics(f["id"]) for f in finished
            ]
            stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

            for fixture, stats in zip(finished, stats_list):
                entry: dict[str, Any] = {"match": fixture}
                if isinstance(stats, Exception):
                    logger.warning(
                        "Stats fetch failed for fixture %d: %s",
                        fixture["id"],
                        stats,
                    )
                elif stats is not None:
                    entry["stats"] = stats
                results.append(entry)

        logger.info("Bulk ingestion complete. Total records: %d", len(results))
        return results

    # ------------------------------------------------------------------
    # Normalisers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_fixture(raw: dict) -> dict:
        """Transform a raw API-Football fixture dict into DB-compatible format.

        Args:
            raw: Single element from the API 'response' list.

        Returns:
            Dict with keys matching the `matches` table columns.
        """
        fixture = raw.get("fixture", {})
        teams = raw.get("teams", {})
        goals = raw.get("goals", {})
        league = raw.get("league", {})

        return {
            "id": fixture.get("id"),
            "season": league.get("season"),
            "round": league.get("round"),
            "match_date": fixture.get("date"),
            "home_team_id": teams.get("home", {}).get("id"),
            "away_team_id": teams.get("away", {}).get("id"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "status": fixture.get("status", {}).get("short"),
            "venue": fixture.get("venue", {}).get("name"),
        }

    @staticmethod
    def _normalise_statistics(fixture_id: int, raw: list[dict]) -> dict:
        """Transform raw per-team statistics into a `match_stats`-compatible dict.

        Args:
            fixture_id: The fixture this stats payload belongs to.
            raw: 'response' list (two elements: home then away team stats).

        Returns:
            Dict with keys matching the `match_stats` table columns.
        """

        def _stat(team_data: dict, key: str) -> int | float | None:
            for item in team_data.get("statistics", []):
                if item.get("type") == key:
                    return item.get("value")
            return None

        home_data = raw[0] if len(raw) > 0 else {}
        away_data = raw[1] if len(raw) > 1 else {}

        return {
            "match_id": fixture_id,
            "home_shots": _stat(home_data, "Total Shots"),
            "away_shots": _stat(away_data, "Total Shots"),
            "home_shots_on_target": _stat(home_data, "Shots on Goal"),
            "away_shots_on_target": _stat(away_data, "Shots on Goal"),
            "home_possession": _stat(home_data, "Ball Possession"),
            "away_possession": _stat(away_data, "Ball Possession"),
            "home_xg": _stat(home_data, "expected_goals"),
            "away_xg": _stat(away_data, "expected_goals"),
        }
