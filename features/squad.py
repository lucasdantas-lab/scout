"""Squad strength calculations based on match lineups and player ratings.

Computes a weighted squad quality index per match for each team,
useful both for historical analysis and pre-match prediction with
estimated lineups.
"""

import logging
from typing import Any

import pandas as pd

from config import SQUAD_WEIGHT_STARTER, SQUAD_WEIGHT_SUB

logger = logging.getLogger(__name__)


def compute_squad_strength(
    match_id: int,
    team_id: int,
    lineups_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> float:
    """Compute a weighted squad quality index for a team in a match.

    The index is calculated as::

        strength = Σ(rating × weight) / 11

    where weight = SQUAD_WEIGHT_STARTER for starters and
    weight = SQUAD_WEIGHT_SUB × (minutes / 90) for substitutes.
    The result is normalised by the team's historical average rating.

    Args:
        match_id: The fixture id.
        team_id: The team id.
        lineups_df: DataFrame with columns: match_id, team_id, player_id,
            is_starter, minutes_played, rating.
        players_df: DataFrame with columns: id, team_id, overall_rating.

    Returns:
        Float where 1.0 = full-strength squad, < 1.0 = weakened.
    """
    lineup = lineups_df[
        (lineups_df["match_id"] == match_id)
        & (lineups_df["team_id"] == team_id)
    ]

    if lineup.empty:
        return 1.0

    # Merge player ratings
    players_lookup = players_df.set_index("id")["overall_rating"].to_dict()

    weighted_sum = 0.0
    weight_total = 0.0

    for _, row in lineup.iterrows():
        pid = row["player_id"]
        rating = row.get("rating")
        if rating is None or pd.isna(rating):
            rating = players_lookup.get(pid, 6.5)

        minutes = row.get("minutes_played", 0) or 0
        is_starter = row.get("is_starter", False)

        if is_starter:
            w = SQUAD_WEIGHT_STARTER
        else:
            w = SQUAD_WEIGHT_SUB * (minutes / 90.0) if minutes > 0 else 0.0

        weighted_sum += float(rating) * w
        weight_total += w

    if weight_total == 0:
        return 1.0

    match_strength = weighted_sum / 11.0

    # Normalise by team's historical average rating
    team_players = players_df[players_df["team_id"] == team_id]
    if not team_players.empty:
        avg_rating = team_players["overall_rating"].mean()
        if avg_rating > 0:
            return round(match_strength / avg_rating, 4)

    return round(match_strength / 6.5, 4)


def build_squad_features(
    matches_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add home_squad and away_squad columns to a match DataFrame.

    Args:
        matches_df: DataFrame with at least id, home_team_id, away_team_id.
        lineups_df: Full lineups DataFrame across all matches.
        players_df: Full players DataFrame.

    Returns:
        Copy of matches_df with home_squad and away_squad columns.
    """
    df = matches_df.copy()
    home_squads: list[float] = []
    away_squads: list[float] = []

    for _, row in df.iterrows():
        mid = row["id"]
        home_squads.append(
            compute_squad_strength(mid, row["home_team_id"], lineups_df, players_df)
        )
        away_squads.append(
            compute_squad_strength(mid, row["away_team_id"], lineups_df, players_df)
        )

    df["home_squad"] = home_squads
    df["away_squad"] = away_squads

    logger.info("build_squad_features: added squad columns to %d rows.", len(df))
    return df


def estimate_squad_for_upcoming(
    upcoming_match: dict[str, Any],
    context_json: dict[str, Any] | None,
) -> tuple[float, float]:
    """Estimate squad strength for a match not yet played.

    Uses the processed context from the context_agent to penalise
    for confirmed absences and uncertain availability.

    Args:
        upcoming_match: Dict with home_team_id, away_team_id.
        context_json: Processed context from context_agent, with keys
            'home' and 'away' each containing 'ausencias_confirmadas',
            'duvidas', 'confianca'.

    Returns:
        Tuple of (home_squad_estimate, away_squad_estimate) where 1.0
        is full strength. Also provides an implicit uncertainty range:
        the lower bound assumes all doubts are absent; the upper bound
        assumes all doubts play.
    """
    if not context_json:
        return 1.0, 1.0

    def _estimate_side(side_ctx: dict[str, Any]) -> float:
        if not side_ctx:
            return 1.0
        n_absent = len(side_ctx.get("ausencias_confirmadas", []))
        n_doubt = len(side_ctx.get("duvidas", []))
        confidence = side_ctx.get("confianca", 0.5)

        # Each confirmed absence reduces by ~3%, each doubt by ~1.5%
        # weighted by the confidence of the context information
        penalty = (n_absent * 0.03 + n_doubt * 0.015) * confidence
        return round(max(1.0 - penalty, 0.7), 4)

    home_est = _estimate_side(context_json.get("home", {}))
    away_est = _estimate_side(context_json.get("away", {}))

    return home_est, away_est
