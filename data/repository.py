"""Database access layer via supabase-py.

Provides two repository classes:
- MatchRepository  — CRUD for teams, matches, and match statistics.
- ModelRepository  — CRUD for model parameters and predictions.
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
            team_dict: Dict with keys: id, name, short_name, city.
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
            match_dict: Dict compatible with the `matches` table columns.
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
            stats_dict: Dict compatible with the `match_stats` table columns.
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

    def get_finished_matches(self, season: int | None = None) -> pd.DataFrame:
        """Retrieve all finished matches, optionally filtered by season.

        Args:
            season: If provided, only return matches for this season year.

        Returns:
            DataFrame with matches table columns.
        """
        try:
            query = (
                self._client.table("matches")
                .select("*")
                .eq("status", "FT")
            )
            if season is not None:
                query = query.eq("season", season)
            response = query.execute()
            return pd.DataFrame(response.data)
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
        parameter_type: str = "posterior_mean",
    ) -> None:
        """Persist model parameters for a single team.

        Args:
            run_id: Identifier for this model run (stored in model_version).
            team_id: Team's API-Football id.
            attack: Attack strength parameter.
            defense: Defense strength parameter.
            season: Season year these parameters were estimated from.
            parameter_type: 'posterior_mean' or 'posterior_std'.
        """
        record = {
            "season": season,
            "team_id": team_id,
            "attack": float(attack),
            "defense": float(defense),
            "parameter_type": parameter_type,
        }
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
            prediction_dict: Dict compatible with the `predictions` table.
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

    def get_latest_predictions(self) -> pd.DataFrame:
        """Retrieve all predictions, most recent first.

        Returns:
            DataFrame with predictions table columns.
        """
        try:
            response = (
                self._client.table("predictions")
                .select("*")
                .order("generated_at", desc=True)
                .execute()
            )
            return pd.DataFrame(response.data)
        except Exception as exc:
            logger.error("Failed to fetch latest predictions: %s", exc)
            raise

    def get_predictions_with_results(self) -> pd.DataFrame:
        """Retrieve predictions that have a corresponding finished match.

        Joins predictions with matches to include actual results.

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
            rows = []
            for item in response.data:
                flat = {**item}
                match_data = flat.pop("matches", {})
                flat.update(match_data)
                rows.append(flat)
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.error("Failed to fetch predictions with results: %s", exc)
            raise
