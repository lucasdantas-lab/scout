"""Recent-form calculation with exponential time decay.

Provides per-team form scores and adds them as features to a match DataFrame
without leaking future information.
"""

import logging

import numpy as np
import pandas as pd

from config import DECAY_RATE

logger = logging.getLogger(__name__)

_POINTS = {1: 3, 0: 1, -1: 0}  # result code → points


def _match_result(goals_for: int, goals_against: int) -> int:
    """Return a result code: 1=win, 0=draw, -1=loss."""
    if goals_for > goals_against:
        return 1
    if goals_for == goals_against:
        return 0
    return -1


def compute_form(
    matches_df: pd.DataFrame,
    team_id: int,
    n_games: int = 6,
    decay: float = DECAY_RATE,
    before_date: pd.Timestamp | None = None,
) -> float:
    """Compute a team's recent form score.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date.
        team_id: The team whose form is being computed.
        n_games: Number of most recent games to consider.
        decay: Exponential decay factor; applied as decay^i where i=0 is the
            most recent game.
        before_date: If provided, only consider games strictly before this
            timestamp (prevents data leakage).

    Returns:
        Form score normalised to [0, 1]. Returns 0.5 when the team has no
        recent matches.
    """
    df = matches_df.dropna(
        subset=["home_goals", "away_goals", "match_date"]
    ).copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)

    team_mask = (df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)
    team_games = df[team_mask].sort_values("match_date", ascending=False)

    if before_date is not None:
        before_date = pd.Timestamp(before_date).tz_localize("UTC") if before_date.tzinfo is None else before_date  # type: ignore[union-attr]
        team_games = team_games[team_games["match_date"] < before_date]

    team_games = team_games.head(n_games)

    if team_games.empty:
        return 0.5

    total_weight = 0.0
    weighted_points = 0.0
    max_points_per_game = 3.0

    for i, (_, row) in enumerate(team_games.iterrows()):
        if row["home_team_id"] == team_id:
            goals_for = int(row["home_goals"])
            goals_against = int(row["away_goals"])
        else:
            goals_for = int(row["away_goals"])
            goals_against = int(row["home_goals"])

        result_code = _match_result(goals_for, goals_against)
        pts = _POINTS[result_code]
        weight = decay**i
        weighted_points += pts * weight
        total_weight += max_points_per_game * weight

    return weighted_points / total_weight if total_weight > 0 else 0.5


def build_form_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add home_form and away_form columns to a match DataFrame.

    For each match, form is computed using only matches that occurred
    **before** that match's date, preventing data leakage.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date.

    Returns:
        Copy of matches_df with two additional columns:
        - home_form: form score for the home team in [0, 1]
        - away_form: form score for the away team in [0, 1]
    """
    df = matches_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    df = df.sort_values("match_date").reset_index(drop=True)

    home_forms: list[float] = []
    away_forms: list[float] = []

    for _, row in df.iterrows():
        cutoff: pd.Timestamp = row["match_date"]
        home_forms.append(
            compute_form(df, team_id=row["home_team_id"], before_date=cutoff)
        )
        away_forms.append(
            compute_form(df, team_id=row["away_team_id"], before_date=cutoff)
        )

    df["home_form"] = home_forms
    df["away_form"] = away_forms
    logger.info("build_form_features: added form columns to %d rows.", len(df))
    return df
