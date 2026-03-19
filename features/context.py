"""Contextual features specific to the Brasileirão Série A.

Computes fatigue, match importance, and altitude adjustments.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Cities / venues where altitude is a significant factor for visiting teams
_HIGH_ALTITUDE_CITIES = {
    "cuiabá",
    "brasília",
    "goiânia",
    "goiania",
    "distrito federal",
    "df",
    "go",
    "mt",
}

_ALTITUDE_FACTOR = 1.08  # penalty applied to visiting team at high-altitude venues


def compute_fatigue(
    matches_df: pd.DataFrame,
    team_id: int,
    match_date: pd.Timestamp,
    window_days: int = 21,
) -> int:
    """Count how many matches a team played in the preceding window.

    Args:
        matches_df: Full match DataFrame with match_date, home_team_id,
            away_team_id columns.
        team_id: Team to evaluate.
        match_date: Reference date (the upcoming match). Games on or after
            this date are excluded.
        window_days: Look-back period in days.

    Returns:
        Integer count of matches in [match_date - window_days, match_date).
    """
    match_date = pd.Timestamp(match_date)
    if match_date.tzinfo is None:
        match_date = match_date.tz_localize("UTC")

    df = matches_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)

    window_start = match_date - pd.Timedelta(days=window_days)
    mask = (
        ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id))
        & (df["match_date"] >= window_start)
        & (df["match_date"] < match_date)
    )
    return int(mask.sum())


def compute_match_importance(
    standings_df: pd.DataFrame,
    team_id: int,
    round_num: int,
    total_rounds: int = 38,
) -> float:
    """Estimate the importance of a match for a given team.

    Importance is high for teams fighting for the title or against relegation,
    and lower for mid-table sides with no clear objective.

    Args:
        standings_df: DataFrame with columns: team_id, position, points.
            Must contain all teams currently in the standings.
        team_id: Team to evaluate.
        round_num: Current round number.
        total_rounds: Total rounds in the season (38 for Série A).

    Returns:
        Importance score in [0, 1].
    """
    n_teams = len(standings_df)
    if n_teams == 0:
        return 0.5

    team_row = standings_df[standings_df["team_id"] == team_id]
    if team_row.empty:
        logger.warning(
            "compute_match_importance: team_id %s not found in standings.", team_id
        )
        return 0.5

    position = int(team_row["position"].iloc[0])
    rounds_remaining = max(total_rounds - round_num, 1)
    progress = round_num / total_rounds  # 0 → 1 across the season

    # Zones
    title_zone = position <= 4           # Libertadores
    relegation_zone = position >= n_teams - 3  # Bottom 4 go down

    if title_zone or relegation_zone:
        zone_score = 1.0
    elif position <= 6:
        zone_score = 0.7  # Sul-Americana / Copa do Brasil spots
    elif position <= n_teams // 2:
        zone_score = 0.4
    else:
        zone_score = 0.3

    # Urgency increases as the season progresses
    urgency = 0.5 + 0.5 * progress

    importance = min(zone_score * urgency, 1.0)
    return round(importance, 4)


def _venue_is_high_altitude(venue: str | None) -> bool:
    """Return True when the venue string indicates a high-altitude city."""
    if not venue:
        return False
    venue_lower = venue.lower()
    return any(city in venue_lower for city in _HIGH_ALTITUDE_CITIES)


def build_context_features(
    matches_df: pd.DataFrame,
    standings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add contextual feature columns to a match DataFrame.

    Adds: home_fatigue, away_fatigue, home_importance, away_importance,
    altitude_factor.

    Args:
        matches_df: Match DataFrame with columns: match_date, home_team_id,
            away_team_id, round (optional), venue (optional).
        standings_df: Optional standings DataFrame (team_id, position, points).
            When None, importance scores default to 0.5.

    Returns:
        Copy of matches_df with added context columns.
    """
    df = matches_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)

    home_fatigue_list: list[int] = []
    away_fatigue_list: list[int] = []
    home_importance_list: list[float] = []
    away_importance_list: list[float] = []
    altitude_list: list[float] = []

    for _, row in df.iterrows():
        match_date: pd.Timestamp = row["match_date"]
        home_id: int = row["home_team_id"]
        away_id: int = row["away_team_id"]

        home_fatigue_list.append(compute_fatigue(df, home_id, match_date))
        away_fatigue_list.append(compute_fatigue(df, away_id, match_date))

        if standings_df is not None and "round" in row and pd.notna(row.get("round")):
            # Extract numeric round
            try:
                rnd = int(str(row["round"]).split()[-1])
            except (ValueError, IndexError):
                rnd = 1
            home_importance_list.append(
                compute_match_importance(standings_df, home_id, rnd)
            )
            away_importance_list.append(
                compute_match_importance(standings_df, away_id, rnd)
            )
        else:
            home_importance_list.append(0.5)
            away_importance_list.append(0.5)

        venue = row.get("venue")
        altitude_list.append(
            _ALTITUDE_FACTOR if _venue_is_high_altitude(venue) else 1.0
        )

    df["home_fatigue"] = home_fatigue_list
    df["away_fatigue"] = away_fatigue_list
    df["home_importance"] = home_importance_list
    df["away_importance"] = away_importance_list
    df["altitude_factor"] = altitude_list

    logger.info(
        "build_context_features: added context columns to %d rows.", len(df)
    )
    return df
