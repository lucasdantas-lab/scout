"""Async client for the API-Football REST API.

Handles fixture fetching, per-fixture statistics, lineups, events,
player ratings, upcoming fixtures, and bulk ingestion with rate
limiting and exponential back-off.
"""

import asyncio
import logging
from typing import Any

import httpx
from tqdm import tqdm

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
    ) -> tuple[list[dict], list[dict]]:
        """Fetch all fixtures for a given season and league.

        Args:
            season: Four-digit season year (e.g. 2023).
            league_id: API-Football league identifier.

        Returns:
            Tuple of (matches, teams) where matches is a list of dicts
            compatible with the ``matches`` table and teams is a list of
            dicts compatible with the ``teams`` table.
        """
        raw = await self._get(
            "/fixtures", {"league": league_id, "season": season}
        )
        matches = [self._normalise_fixture(f) for f in raw]
        teams_by_id: dict[int, dict] = {}
        for f in raw:
            for side in ("home", "away"):
                t = f.get("teams", {}).get(side, {})
                tid = t.get("id")
                if tid and tid not in teams_by_id:
                    teams_by_id[tid] = self._normalise_team(t)
        return matches, list(teams_by_id.values())

    async def fetch_statistics(self, fixture_id: int) -> dict | None:
        """Fetch per-fixture statistics.

        Args:
            fixture_id: API-Football fixture id.

        Returns:
            Dict compatible with the ``match_stats`` table, or None when no
            statistics are available for this fixture.
        """
        raw = await self._get("/fixtures/statistics", {"fixture": fixture_id})
        if not raw:
            return None
        return self._normalise_statistics(fixture_id, raw)

    async def fetch_lineups(self, fixture_id: int) -> list[dict]:
        """Fetch confirmed lineups for a fixture.

        Args:
            fixture_id: API-Football fixture id.

        Returns:
            List of dicts compatible with the ``match_lineups`` table.
        """
        raw = await self._get("/fixtures/lineups", {"fixture": fixture_id})
        if not raw:
            return []
        return self._normalise_lineups(fixture_id, raw)

    async def fetch_events(self, fixture_id: int) -> list[dict]:
        """Fetch match events (goals, cards, substitutions).

        Args:
            fixture_id: API-Football fixture id.

        Returns:
            List of dicts compatible with the ``match_events`` table.
        """
        raw = await self._get("/fixtures/events", {"fixture": fixture_id})
        if not raw:
            return []
        return self._normalise_events(fixture_id, raw)

    async def fetch_players(self, fixture_id: int) -> list[dict]:
        """Fetch per-player ratings for a fixture.

        Args:
            fixture_id: API-Football fixture id.

        Returns:
            List of dicts compatible with the ``players`` table
            plus per-match ``rating`` and ``minutes_played`` fields.
        """
        raw = await self._get("/fixtures/players", {"fixture": fixture_id})
        if not raw:
            return []
        return self._normalise_players(fixture_id, raw)

    async def fetch_upcoming(
        self, league_id: int = LEAGUE_ID, next_n: int = 10
    ) -> tuple[list[dict], list[dict]]:
        """Fetch the next N scheduled (not yet played) fixtures.

        Args:
            league_id: API-Football league identifier.
            next_n: Maximum number of upcoming fixtures to retrieve.

        Returns:
            Tuple of (matches, teams).
        """
        raw = await self._get(
            "/fixtures", {"league": league_id, "next": next_n}
        )
        matches = [self._normalise_fixture(f) for f in raw]
        teams_by_id: dict[int, dict] = {}
        for f in raw:
            for side in ("home", "away"):
                t = f.get("teams", {}).get(side, {})
                tid = t.get("id")
                if tid and tid not in teams_by_id:
                    teams_by_id[tid] = self._normalise_team(t)
        return matches, list(teams_by_id.values())

    async def bulk_ingest(
        self, seasons: list[int]
    ) -> tuple[list[dict], list[dict]]:
        """Orchestrate full ingestion for multiple seasons.

        For every finished fixture (status='FT') fetches statistics,
        lineups, events, and player ratings in parallel via
        ``asyncio.gather``. Progress is displayed with ``tqdm``.

        Args:
            seasons: List of season years to ingest.

        Returns:
            Tuple of (results, teams) where results is a list of dicts
            containing 'match' and optionally 'stats', 'lineups',
            'events', 'players' keys.
        """
        results: list[dict] = []
        all_teams: dict[int, dict] = {}

        for season in seasons:
            logger.info("Ingesting season %d …", season)
            fixtures, teams = await self.fetch_fixtures(season)
            logger.info(
                "Season %d: %d fixtures found.", season, len(fixtures)
            )

            for t in teams:
                all_teams[t["id"]] = t

            finished = [f for f in fixtures if f.get("status") == "FT"]
            logger.info(
                "Season %d: %d finished fixtures.", season, len(finished)
            )

            pbar = tqdm(
                total=len(finished),
                desc=f"Season {season}",
                unit="match",
            )

            async def _fetch_all_for_fixture(fixture: dict) -> dict:
                """Fetch stats, lineups, events, players for one fixture."""
                fid = fixture["id"]
                stats_coro = self.fetch_statistics(fid)
                lineups_coro = self.fetch_lineups(fid)
                events_coro = self.fetch_events(fid)
                players_coro = self.fetch_players(fid)

                gathered = await asyncio.gather(
                    stats_coro, lineups_coro, events_coro, players_coro,
                    return_exceptions=True,
                )
                entry: dict[str, Any] = {"match": fixture}

                labels = ("stats", "lineups", "events", "players")
                for label, result in zip(labels, gathered):
                    if isinstance(result, Exception):
                        logger.warning(
                            "%s fetch failed for fixture %d: %s",
                            label, fid, result,
                        )
                    elif result:
                        entry[label] = result

                pbar.update(1)
                return entry

            tasks = [_fetch_all_for_fixture(f) for f in finished]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            pbar.close()

        logger.info(
            "Bulk ingestion complete. Teams: %d | Fixtures: %d",
            len(all_teams),
            len(results),
        )
        return results, list(all_teams.values())

    # ------------------------------------------------------------------
    # Normalisers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_team(raw: dict) -> dict:
        """Transform a raw API-Football team dict into DB-compatible format.

        Args:
            raw: Team object from the 'teams.home' or 'teams.away' field.

        Returns:
            Dict with keys matching the ``teams`` table columns.
        """
        return {
            "id": raw.get("id"),
            "name": raw.get("name"),
            "short_name": raw.get("code"),
            "city": None,
        }

    @staticmethod
    def _normalise_fixture(raw: dict) -> dict:
        """Transform a raw API-Football fixture dict into DB-compatible format.

        Args:
            raw: Single element from the API 'response' list.

        Returns:
            Dict with keys matching the ``matches`` table columns.
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
            "home_xg": None,   # populated from statistics if available
            "away_xg": None,
            "status": fixture.get("status", {}).get("short"),
            "venue": fixture.get("venue", {}).get("name"),
        }

    @staticmethod
    def _normalise_statistics(fixture_id: int, raw: list[dict]) -> dict:
        """Transform raw per-team statistics into a ``match_stats``-compatible dict.

        Args:
            fixture_id: The fixture this stats payload belongs to.
            raw: 'response' list (two elements: home then away team stats).

        Returns:
            Dict with keys matching the ``match_stats`` table columns.
        """

        def _stat(team_data: dict, key: str) -> int | float | None:
            for item in team_data.get("statistics", []):
                if item.get("type") == key:
                    return item.get("value")
            return None

        home_data = raw[0] if len(raw) > 0 else {}
        away_data = raw[1] if len(raw) > 1 else {}

        def _pct(value: int | float | str | None) -> float | None:
            """Strip trailing '%' and convert to float."""
            if value is None:
                return None
            return float(str(value).rstrip("%"))

        return {
            "match_id": fixture_id,
            "home_shots": _stat(home_data, "Total Shots"),
            "away_shots": _stat(away_data, "Total Shots"),
            "home_shots_on_target": _stat(home_data, "Shots on Goal"),
            "away_shots_on_target": _stat(away_data, "Shots on Goal"),
            "home_possession": _pct(_stat(home_data, "Ball Possession")),
            "away_possession": _pct(_stat(away_data, "Ball Possession")),
            "home_xg": _stat(home_data, "expected_goals"),
            "away_xg": _stat(away_data, "expected_goals"),
        }

    @staticmethod
    def _normalise_lineups(fixture_id: int, raw: list[dict]) -> list[dict]:
        """Transform raw lineups into ``match_lineups``-compatible dicts.

        Args:
            fixture_id: The fixture this lineup belongs to.
            raw: 'response' list with one entry per team.

        Returns:
            List of dicts for the ``match_lineups`` table.
        """
        rows: list[dict] = []
        for team_entry in raw:
            team_id = team_entry.get("team", {}).get("id")
            if not team_id:
                continue

            for player in team_entry.get("startXI", []):
                p = player.get("player", {})
                rows.append({
                    "match_id": fixture_id,
                    "team_id": team_id,
                    "player_id": p.get("id"),
                    "is_starter": True,
                    "minutes_played": 90,
                    "rating": None,
                })

            for player in team_entry.get("substitutes", []):
                p = player.get("player", {})
                rows.append({
                    "match_id": fixture_id,
                    "team_id": team_id,
                    "player_id": p.get("id"),
                    "is_starter": False,
                    "minutes_played": 0,
                    "rating": None,
                })

        return rows

    @staticmethod
    def _normalise_events(fixture_id: int, raw: list[dict]) -> list[dict]:
        """Transform raw events into ``match_events``-compatible dicts.

        Args:
            fixture_id: The fixture these events belong to.
            raw: 'response' list with one entry per event.

        Returns:
            List of dicts for the ``match_events`` table.
        """
        type_map = {
            "Goal": "goal",
            "Card": "yellow_card",
            "subst": "substitution",
        }
        rows: list[dict] = []
        for event in raw:
            etype_raw = event.get("type", "")
            detail = event.get("detail", "")
            etype = type_map.get(etype_raw, etype_raw.lower())

            if etype_raw == "Card" and "Red" in detail:
                etype = "red_card"

            rows.append({
                "match_id": fixture_id,
                "team_id": event.get("team", {}).get("id"),
                "event_type": etype,
                "minute": event.get("time", {}).get("elapsed"),
                "player_id": event.get("player", {}).get("id"),
            })
        return rows

    @staticmethod
    def _normalise_players(fixture_id: int, raw: list[dict]) -> list[dict]:
        """Transform raw player stats into enriched player dicts.

        Returns player data plus per-match ``rating`` and ``minutes_played``
        fields suitable for both ``players`` and ``match_lineups`` updates.

        Args:
            fixture_id: The fixture these player stats belong to.
            raw: 'response' list with one entry per team.

        Returns:
            List of dicts with player info and per-match performance.
        """
        rows: list[dict] = []
        for team_entry in raw:
            team_id = team_entry.get("team", {}).get("id")
            if not team_id:
                continue
            for player_wrap in team_entry.get("players", []):
                p = player_wrap.get("player", {})
                stats_list = player_wrap.get("statistics", [])
                stats = stats_list[0] if stats_list else {}
                games = stats.get("games", {})

                rating_str = games.get("rating")
                rating = float(rating_str) if rating_str else None
                minutes = games.get("minutes")
                position = games.get("position")

                rows.append({
                    "fixture_id": fixture_id,
                    "team_id": team_id,
                    "player_id": p.get("id"),
                    "player_name": p.get("name"),
                    "position": position,
                    "rating": rating,
                    "minutes_played": minutes,
                })
        return rows
