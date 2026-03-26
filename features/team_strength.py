"""Team attack and defence strength calculations.

Provides raw and exponentially time-weighted strength metrics derived
from historical match results, with optional xG-based computation.
"""

import logging

import numpy as np
import pandas as pd

from config import DECAY_RATE

logger = logging.getLogger(__name__)


def _goal_columns(df: pd.DataFrame, use_xg: bool) -> tuple[str, str]:
    """Return the column names to use as goal proxies.

    Falls back to actual goals when xG is not available.

    Args:
        df: Match DataFrame.
        use_xg: Whether to prefer xG columns.

    Returns:
        Tuple of (home_col, away_col).
    """
    if use_xg and "home_xg" in df.columns and "away_xg" in df.columns:
        has_xg = df["home_xg"].notna().sum() > 0
        if has_xg:
            return "home_xg", "away_xg"
    return "home_goals", "away_goals"


def compute_raw_strength(
    matches_df: pd.DataFrame,
    use_xg: bool = False,
) -> pd.DataFrame:
    """Compute raw attack/defence strength ratios for each team.

    Strength is expressed as the ratio of a team's goals-per-game to the
    league average, split by home/away role. When ``use_xg`` is True and
    xG data is available, xG is used instead of actual goals.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals,
            and optionally home_xg, away_xg.
        use_xg: If True, prefer xG over actual goals.

    Returns:
        DataFrame with columns: team_id, attack_home, attack_away,
        defense_home, defense_away, xg_attack, xg_defense.
    """
    df = matches_df.dropna(subset=["home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)

    h_col, a_col = _goal_columns(df, use_xg)

    # Ensure chosen columns are float
    df[h_col] = df[h_col].astype(float)
    df[a_col] = df[a_col].astype(float)

    # Fill NaN xG with actual goals for mixed datasets
    if h_col == "home_xg":
        df[h_col] = df[h_col].fillna(df["home_goals"])
        df[a_col] = df[a_col].fillna(df["away_goals"])

    league_avg_scored = (df[h_col].sum() + df[a_col].sum()) / (2 * len(df))
    if league_avg_scored == 0:
        raise ValueError("League average goals is zero — check input data.")

    # Also compute xG averages for reference columns
    has_xg_data = "home_xg" in df.columns and df["home_xg"].notna().any()
    if has_xg_data:
        df["_hxg"] = df["home_xg"].astype(float).fillna(df["home_goals"])
        df["_axg"] = df["away_xg"].astype(float).fillna(df["away_goals"])
        xg_avg = (df["_hxg"].sum() + df["_axg"].sum()) / (2 * len(df))
    else:
        xg_avg = None

    records: dict[int, dict] = {}

    for team_id in set(df["home_team_id"]).union(df["away_team_id"]):
        home_matches = df[df["home_team_id"] == team_id]
        away_matches = df[df["away_team_id"] == team_id]

        gs_home = (
            home_matches[h_col].mean()
            if len(home_matches) > 0
            else league_avg_scored
        )
        gc_home = (
            home_matches[a_col].mean()
            if len(home_matches) > 0
            else league_avg_scored
        )
        gs_away = (
            away_matches[a_col].mean()
            if len(away_matches) > 0
            else league_avg_scored
        )
        gc_away = (
            away_matches[h_col].mean()
            if len(away_matches) > 0
            else league_avg_scored
        )

        rec: dict = {
            "team_id": team_id,
            "attack_home": gs_home / league_avg_scored,
            "attack_away": gs_away / league_avg_scored,
            "defense_home": gc_home / league_avg_scored,
            "defense_away": gc_away / league_avg_scored,
        }

        # xG-based attack/defense (reference, always computed when data exists)
        if xg_avg and xg_avg > 0:
            xg_scored = (
                home_matches["_hxg"].sum() + away_matches["_axg"].sum()
            )
            xg_conceded = (
                home_matches["_axg"].sum() + away_matches["_hxg"].sum()
            )
            n_games = len(home_matches) + len(away_matches)
            rec["xg_attack"] = (
                (xg_scored / n_games) / xg_avg if n_games > 0 else 1.0
            )
            rec["xg_defense"] = (
                (xg_conceded / n_games) / xg_avg if n_games > 0 else 1.0
            )
        else:
            rec["xg_attack"] = None
            rec["xg_defense"] = None

        records[team_id] = rec

    result = pd.DataFrame(records.values())
    logger.info("compute_raw_strength: computed strength for %d teams.", len(result))
    return result


def compute_weighted_strength(
    matches_df: pd.DataFrame,
    decay_rate: float = DECAY_RATE,
    use_xg: bool = False,
) -> pd.DataFrame:
    """Compute exponentially time-weighted attack/defence strength.

    More recent matches receive higher weight. Matches are ordered from
    oldest to newest; the most recent match has weight = 1.0.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date,
            and optionally home_xg, away_xg.
        decay_rate: Per-match exponential decay factor (0 < decay_rate <= 1).
        use_xg: If True, prefer xG over actual goals.

    Returns:
        DataFrame with columns: team_id, attack_home, attack_away,
        defense_home, defense_away.
    """
    df = matches_df.dropna(subset=["home_goals", "away_goals", "match_date"]).copy()
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    df = df.sort_values("match_date").reset_index(drop=True)

    h_col, a_col = _goal_columns(df, use_xg)
    df[h_col] = df[h_col].astype(float)
    df[a_col] = df[a_col].astype(float)
    if h_col == "home_xg":
        df[h_col] = df[h_col].fillna(df["home_goals"])
        df[a_col] = df[a_col].fillna(df["away_goals"])

    # Weight = decay_rate ^ (n_matches_ago)
    n = len(df)
    df["weight"] = decay_rate ** np.arange(n - 1, -1, -1, dtype=float)

    total_weight = df["weight"].sum()
    league_avg_scored = (
        (df[h_col] * df["weight"]).sum()
        + (df[a_col] * df["weight"]).sum()
    ) / (2 * total_weight)

    if league_avg_scored == 0:
        raise ValueError("Weighted league average goals is zero — check input data.")

    records: dict[int, dict] = {}

    def _wavg(values: pd.Series, weights: pd.Series, fallback: float) -> float:
        total = weights.sum()
        return (values * weights).sum() / total if total > 0 else fallback

    all_team_ids = set(df["home_team_id"]).union(df["away_team_id"])
    for team_id in all_team_ids:
        home_mask = df["home_team_id"] == team_id
        away_mask = df["away_team_id"] == team_id

        gs_home = _wavg(
            df.loc[home_mask, h_col],
            df.loc[home_mask, "weight"],
            league_avg_scored,
        )
        gc_home = _wavg(
            df.loc[home_mask, a_col],
            df.loc[home_mask, "weight"],
            league_avg_scored,
        )
        gs_away = _wavg(
            df.loc[away_mask, a_col],
            df.loc[away_mask, "weight"],
            league_avg_scored,
        )
        gc_away = _wavg(
            df.loc[away_mask, h_col],
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
