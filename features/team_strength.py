"""Team attack and defence strength calculations.

Provides raw and exponentially time-weighted strength metrics derived
from historical match results.
"""

import logging

import numpy as np
import pandas as pd

from config import DECAY_RATE

logger = logging.getLogger(__name__)


def compute_raw_strength(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw attack/defence strength ratios for each team.

    Strength is expressed as the ratio of a team's goals-per-game to the
    league average, split by home/away role.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals.

    Returns:
        DataFrame with columns: team_id, attack_home, attack_away,
        defense_home, defense_away.
    """
    df = matches_df.dropna(subset=["home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)

    league_avg_scored = (df["home_goals"].sum() + df["away_goals"].sum()) / (
        2 * len(df)
    )
    if league_avg_scored == 0:
        raise ValueError("League average goals is zero — check input data.")

    records: dict[int, dict] = {}

    for team_id in set(df["home_team_id"]).union(df["away_team_id"]):
        home_matches = df[df["home_team_id"] == team_id]
        away_matches = df[df["away_team_id"] == team_id]

        gs_home = (
            home_matches["home_goals"].mean()
            if len(home_matches) > 0
            else league_avg_scored
        )
        gc_home = (
            home_matches["away_goals"].mean()
            if len(home_matches) > 0
            else league_avg_scored
        )
        gs_away = (
            away_matches["away_goals"].mean()
            if len(away_matches) > 0
            else league_avg_scored
        )
        gc_away = (
            away_matches["home_goals"].mean()
            if len(away_matches) > 0
            else league_avg_scored
        )

        records[team_id] = {
            "team_id": team_id,
            "attack_home": gs_home / league_avg_scored,
            "attack_away": gs_away / league_avg_scored,
            "defense_home": gc_home / league_avg_scored,
            "defense_away": gc_away / league_avg_scored,
        }

    result = pd.DataFrame(records.values())
    logger.info("compute_raw_strength: computed strength for %d teams.", len(result))
    return result


def compute_weighted_strength(
    matches_df: pd.DataFrame,
    decay_rate: float = DECAY_RATE,
) -> pd.DataFrame:
    """Compute exponentially time-weighted attack/defence strength.

    More recent matches receive higher weight. Matches are ordered from
    oldest to newest; the most recent match has weight = 1.0.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date.
        decay_rate: Per-match exponential decay factor (0 < decay_rate <= 1).

    Returns:
        DataFrame with columns: team_id, attack_home, attack_away,
        defense_home, defense_away.
    """
    df = matches_df.dropna(subset=["home_goals", "away_goals", "match_date"]).copy()
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    df = df.sort_values("match_date").reset_index(drop=True)

    # Weight = decay_rate ^ (n_matches_ago)
    n = len(df)
    df["weight"] = decay_rate ** np.arange(n - 1, -1, -1, dtype=float)

    total_weight = df["weight"].sum()
    league_avg_scored = (
        (df["home_goals"] * df["weight"]).sum()
        + (df["away_goals"] * df["weight"]).sum()
    ) / (2 * total_weight)

    if league_avg_scored == 0:
        raise ValueError("Weighted league average goals is zero — check input data.")

    records: dict[int, dict] = {}

    all_team_ids = set(df["home_team_id"]).union(df["away_team_id"])
    for team_id in all_team_ids:
        home_mask = df["home_team_id"] == team_id
        away_mask = df["away_team_id"] == team_id

        home_w = df.loc[home_mask, "weight"].sum()
        away_w = df.loc[away_mask, "weight"].sum()

        def _wavg(values: pd.Series, weights: pd.Series, fallback: float) -> float:
            total = weights.sum()
            return (values * weights).sum() / total if total > 0 else fallback

        gs_home = _wavg(
            df.loc[home_mask, "home_goals"],
            df.loc[home_mask, "weight"],
            league_avg_scored,
        )
        gc_home = _wavg(
            df.loc[home_mask, "away_goals"],
            df.loc[home_mask, "weight"],
            league_avg_scored,
        )
        gs_away = _wavg(
            df.loc[away_mask, "away_goals"],
            df.loc[away_mask, "weight"],
            league_avg_scored,
        )
        gc_away = _wavg(
            df.loc[away_mask, "home_goals"],
            df.loc[away_mask, "weight"],
            league_avg_scored,
        )

        records[team_id] = {
            "team_id": team_id,
            "attack_home": gs_home / league_avg_scored,
            "attack_away": gs_away / league_avg_scored,
            "defense_home": gc_home / league_avg_scored,
            "defense_away": gc_away / league_avg_scored,
        }

    result = pd.DataFrame(records.values())
    logger.info(
        "compute_weighted_strength: computed for %d teams (decay=%.2f).",
        len(result),
        decay_rate,
    )
    return result
