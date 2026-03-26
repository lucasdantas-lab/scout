"""Database access layer via supabase-py.

Provides two repository classes:
- MatchRepository  — CRUD for teams, matches, match statistics, lineups,
                     events, and players.
- ModelRepository  — CRUD for model parameters, predictions, context,
                     and calibration logs.
"""

import logging
from typing import Any

import pandas as pd
from supabase import Client, create_client

from config import SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)


def _get_client() -> Client:
    """Instantiate and return a Supabase client.

    Returns:
        Authenticated Supabase Client instance.

    Raises:
        ValueError: When SUPABASE_URL or SUPABASE_KEY are not configured.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in the environment."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _paginated_select(
    client: Client,
    table: str,
    select: str = "*",
    filters: dict[str, Any] | None = None,
    page_size: int = 1000,
) -> list[dict]:
    """Fetch all rows from a table using pagination.

    Args:
        client: Supabase client instance.
        table: Table name.
        select: Column selection string.
        filters: Dict of {column: value} equality filters.
        page_size: Number of rows per request.

    Returns:
        List of row dicts.
    """
    all_rows: list[dict] = []
    offset = 0
    while True:
        query = (
            client.table(table)
            .select(select)
            .range(offset, offset + page_size - 1)
        )
        for col, val in (filters or {}).items():
            query = query.eq(col, val)
        batch = query.execute().data
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return all_rows


# ---------------------------------------------------------------------------
# MatchRepository
# ---------------------------------------------------------------------------


class MatchRepository:
    """Provides upsert and query operations for match-related tables."""

    def __init__(self) -> None:
        self._client: Client = _get_client()

    def upsert_team(self, team_dict: dict[str, Any]) -> None:
        """Upsert a single team record.

        Args:
            team_dict: Dict with keys: id, name, short_name, city,
                altitude_factor.
        """
        try:
            self._client.table("teams").upsert(team_dict).execute()
            logger.debug("Upserted team id=%s", team_dict.get("id"))
        except Exception as exc:
            logger.error("Failed to upsert team %s: %s", team_dict.get("id"), exc)
            raise

    def upsert_match(self, match_dict: dict[str, Any]) -> None:
        """Upsert a single match record.

        Args:
            match_dict: Dict compatible with the ``matches`` table columns.
        """
        try:
            self._client.table("matches").upsert(match_dict).execute()
            logger.debug("Upserted match id=%s", match_dict.get("id"))
        except Exception as exc:
            logger.error("Failed to upsert match %s: %s", match_dict.get("id"), exc)
            raise

    def upsert_stats(self, stats_dict: dict[str, Any]) -> None:
        """Upsert match statistics for a single fixture.

        Args:
            stats_dict: Dict compatible with the ``match_stats`` table columns.
        """
        try:
            self._client.table("match_stats").upsert(stats_dict).execute()
            logger.debug(
                "Upserted stats for match_id=%s", stats_dict.get("match_id")
            )
        except Exception as exc:
            logger.error(
                "Failed to upsert stats for match %s: %s",
                stats_dict.get("match_id"),
                exc,
            )
            raise

    def upsert_lineup(self, lineup_dict: dict[str, Any]) -> None:
        """Upsert a single match lineup entry.

        Args:
            lineup_dict: Dict compatible with the ``match_lineups`` table.
        """
        try:
            self._client.table("match_lineups").upsert(
                lineup_dict, on_conflict="match_id,team_id,player_id"
            ).execute()
            logger.debug(
                "Upserted lineup entry match_id=%s player_id=%s",
                lineup_dict.get("match_id"),
                lineup_dict.get("player_id"),
            )
        except Exception as exc:
            logger.error(
                "Failed to upsert lineup for match %s: %s",
                lineup_dict.get("match_id"),
                exc,
            )
            raise

    def upsert_events(self, events_list: list[dict[str, Any]]) -> None:
        """Upsert a batch of match events.

        Args:
            events_list: List of dicts compatible with the ``match_events``
                table columns.
        """
        if not events_list:
            return
        try:
            self._client.table("match_events").upsert(events_list).execute()
            logger.debug(
                "Upserted %d events for match_id=%s",
                len(events_list),
                events_list[0].get("match_id"),
            )
        except Exception as exc:
            logger.error(
                "Failed to upsert events for match %s: %s",
                events_list[0].get("match_id") if events_list else "?",
                exc,
            )
            raise

    def upsert_players(self, players_list: list[dict[str, Any]]) -> None:
        """Upsert a batch of player records.

        Args:
            players_list: List of dicts with at least id, team_id, name,
                position, overall_rating.
        """
        if not players_list:
            return
        try:
            self._client.table("players").upsert(players_list).execute()
            logger.debug("Upserted %d player records.", len(players_list))
        except Exception as exc:
            logger.error("Failed to upsert players: %s", exc)
            raise

    def get_finished_matches(self, season: int | None = None) -> pd.DataFrame:
        """Retrieve all finished matches, optionally filtered by season.

        Args:
            season: If provided, only return matches for this season year.

        Returns:
            DataFrame with matches table columns.
        """
        try:
            filters: dict[str, Any] = {"status": "FT"}
            if season is not None:
                filters["season"] = season
            rows = _paginated_select(self._client, "matches", filters=filters)
            logger.info("get_finished_matches: loaded %d rows.", len(rows))
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.error("Failed to fetch finished matches: %s", exc)
            raise

    def get_upcoming_matches(self) -> pd.DataFrame:
        """Retrieve all scheduled (not yet played) matches.

        Returns:
            DataFrame with matches table columns filtered to status='NS'.
        """
        try:
            response = (
                self._client.table("matches")
                .select("*")
                .eq("status", "NS")
                .execute()
            )
            return pd.DataFrame(response.data)
        except Exception as exc:
            logger.error("Failed to fetch upcoming matches: %s", exc)
            raise

    def get_match_lineups(self, match_id: int) -> pd.DataFrame:
        """Retrieve lineups for a specific match.

        Args:
            match_id: The fixture id.

        Returns:
            DataFrame with match_lineups table columns.
        """
        try:
            response = (
                self._client.table("match_lineups")
                .select("*")
                .eq("match_id", match_id)
                .execute()
            )
            return pd.DataFrame(response.data)
        except Exception as exc:
            logger.error(
                "Failed to fetch lineups for match %s: %s", match_id, exc
            )
            raise


# ---------------------------------------------------------------------------
# ModelRepository
# ---------------------------------------------------------------------------


class ModelRepository:
    """Provides upsert and query operations for model outputs."""

    def __init__(self) -> None:
        self._client: Client = _get_client()

    def save_parameters(
        self,
        run_id: str,
        team_id: int,
        attack: float,
        defense: float,
        season: int,
        attack_std: float | None = None,
        defense_std: float | None = None,
        parameter_type: str = "posterior_mean",
    ) -> None:
        """Persist model parameters for a single team.

        Args:
            run_id: Identifier for this model run.
            team_id: Team's API-Football id.
            attack: Attack strength parameter.
            defense: Defense strength parameter.
            season: Season year these parameters were estimated from.
            attack_std: Posterior std of attack (Bayesian only).
            defense_std: Posterior std of defense (Bayesian only).
            parameter_type: 'posterior_mean' or 'posterior_std'.
        """
        record: dict[str, Any] = {
            "season": season,
            "team_id": team_id,
            "attack": float(attack),
            "defense": float(defense),
            "parameter_type": parameter_type,
        }
        if attack_std is not None:
            record["attack_std"] = float(attack_std)
        if defense_std is not None:
            record["defense_std"] = float(defense_std)

        try:
            self._client.table("model_parameters").insert(record).execute()
            logger.debug(
                "Saved %s parameters for team %s (season %s)",
                parameter_type,
                team_id,
                season,
            )
        except Exception as exc:
            logger.error(
                "Failed to save parameters for team %s: %s", team_id, exc
            )
            raise

    def save_prediction(self, prediction_dict: dict[str, Any]) -> None:
        """Persist a prediction for a single match.

        Args:
            prediction_dict: Dict compatible with the ``predictions`` table.
        """
        try:
            self._client.table("predictions").insert(prediction_dict).execute()
            logger.debug(
                "Saved prediction for match_id=%s",
                prediction_dict.get("match_id"),
            )
        except Exception as exc:
            logger.error(
                "Failed to save prediction for match %s: %s",
                prediction_dict.get("match_id"),
                exc,
            )
            raise

    def save_context(
        self,
        match_id: int,
        raw_news: str,
        processed_context: dict[str, Any],
        model: str,
    ) -> None:
        """Persist context agent output for a match.

        Args:
            match_id: The fixture this context belongs to.
            raw_news: Raw text collected from news sources.
            processed_context: Structured JSON from Claude processing.
            model: Claude model used for processing.
        """
        record = {
            "match_id": match_id,
            "raw_news": raw_news,
            "processed_context": processed_context,
            "agent_model": model,
        }
        try:
            self._client.table("match_context").insert(record).execute()
            logger.debug("Saved context for match_id=%s", match_id)
        except Exception as exc:
            logger.error(
                "Failed to save context for match %s: %s", match_id, exc
            )
            raise

    def save_calibration_log(self, log_dict: dict[str, Any]) -> None:
        """Persist a calibration agent analysis.

        Args:
            log_dict: Dict compatible with the ``calibration_log`` table.
        """
        try:
            self._client.table("calibration_log").insert(log_dict).execute()
            logger.debug(
                "Saved calibration log for round=%s",
                log_dict.get("round_analyzed"),
            )
        except Exception as exc:
            logger.error("Failed to save calibration log: %s", exc)
            raise

    def get_latest_predictions(self) -> pd.DataFrame:
        """Retrieve all predictions joined with match metadata.

        Returns:
            DataFrame with predictions columns plus home_team_id,
            away_team_id, round, match_date, season, home_goals and
            away_goals from matches.
        """
        try:
            all_rows: list[dict] = []
            page_size = 1000
            offset = 0
            while True:
                response = (
                    self._client.table("predictions")
                    .select(
                        "*, matches(home_team_id, away_team_id, round,"
                        " match_date, season, home_goals, away_goals)"
                    )
                    .order("generated_at", desc=True)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                batch = response.data
                for item in batch:
                    flat = {**item}
                    match_data = flat.pop("matches", {}) or {}
                    flat.update(match_data)
                    all_rows.append(flat)
                if len(batch) < page_size:
                    break
                offset += page_size
            logger.info("get_latest_predictions: loaded %d rows.", len(all_rows))
            return pd.DataFrame(all_rows)
        except Exception as exc:
            logger.error("Failed to fetch latest predictions: %s", exc)
            raise

    def get_predictions_with_results(self) -> pd.DataFrame:
        """Retrieve predictions that have a corresponding finished match.

        Returns:
            DataFrame containing prediction columns plus home_goals and
            away_goals from the matches table.
        """
        try:
            response = (
                self._client.table("predictions")
                .select(
                    "*, matches!inner(home_goals, away_goals, status, match_date)"
                )
                .eq("matches.status", "FT")
                .execute()
            )
            rows: list[dict] = []
            for item in response.data:
                flat = {**item}
                match_data = flat.pop("matches", {})
                flat.update(match_data)
                rows.append(flat)
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.error("Failed to fetch predictions with results: %s", exc)
            raise

    def get_calibration_history(self) -> pd.DataFrame:
        """Retrieve all calibration agent logs.

        Returns:
            DataFrame with calibration_log table columns ordered by
            generated_at descending.
        """
        try:
            response = (
                self._client.table("calibration_log")
                .select("*")
                .order("generated_at", desc=True)
                .execute()
            )
            return pd.DataFrame(response.data)
        except Exception as exc:
            logger.error("Failed to fetch calibration history: %s", exc)
            raise

    def get_match_context(self, match_id: int) -> dict[str, Any] | None:
        """Retrieve the most recent context for a match.

        Args:
            match_id: The fixture id.

        Returns:
            The processed_context dict, or None if no context exists.
        """
        try:
            response = (
                self._client.table("match_context")
                .select("*")
                .eq("match_id", match_id)
                .order("generated_at", desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0]
            return None
        except Exception as exc:
            logger.error(
                "Failed to fetch context for match %s: %s", match_id, exc
            )
            raise
