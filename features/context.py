"""Contextual features specific to the Brasileirão Série A.

Computes fatigue (with travel penalty), match importance, and altitude
adjustments.
"""

import logging

import pandas as pd

from config import ALTITUDE_FACTOR

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

# Major Brazilian football cities with approximate latitude/longitude
# Used to estimate travel distances for fatigue computation.
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "são paulo": (-23.55, -46.63),
    "rio de janeiro": (-22.91, -43.17),
    "belo horizonte": (-19.92, -43.94),
    "porto alegre": (-30.03, -51.23),
    "curitiba": (-25.43, -49.27),
    "salvador": (-12.97, -38.51),
    "fortaleza": (-3.72, -38.54),
    "recife": (-8.05, -34.87),
    "brasília": (-15.79, -47.88),
    "goiânia": (-16.69, -49.25),
    "cuiabá": (-15.60, -56.10),
    "florianópolis": (-27.59, -48.55),
    "belém": (-1.46, -48.50),
    "são luís": (-2.53, -44.28),
    "manaus": (-3.12, -60.02),
    "natal": (-5.79, -35.21),
    "maceió": (-9.67, -35.74),
    "aracaju": (-10.91, -37.07),
    "campinas": (-22.91, -47.06),
    "santos": (-23.96, -46.33),
    "londrina": (-23.31, -51.16),
    "caxias do sul": (-29.17, -51.18),
    "chapecó": (-27.10, -52.62),
}

_LONG_TRAVEL_KM = 1000.0
_TRAVEL_FATIGUE_MULTIPLIER = 1.5


def _estimate_distance_km(city_a: str | None, city_b: str | None) -> float:
    """Estimate straight-line distance between two cities in km.

    Args:
        city_a: City name (or venue string).
        city_b: City name (or venue string).

    Returns:
        Approximate distance in km, or 0.0 if either city is unknown.
    """
    import math

    def _find_coords(name: str | None) -> tuple[float, float] | None:
        if not name:
            return None
        name_lower = name.lower()
        for city, coords in _CITY_COORDS.items():
            if city in name_lower:
                return coords
        return None

    coords_a = _find_coords(city_a)
    coords_b = _find_coords(city_b)
    if not coords_a or not coords_b:
        return 0.0

    # Haversine formula
    lat1, lon1 = math.radians(coords_a[0]), math.radians(coords_a[1])
    lat2, lon2 = math.radians(coords_b[0]), math.radians(coords_b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def compute_fatigue(
    matches_df: pd.DataFrame,
    team_id: int,
    match_date: pd.Timestamp,
    window_days: int = 21,
) -> float:
    """Compute fatigue score accounting for match frequency and travel.

    Away games with estimated travel > 1000km count as 1.5 games.

    Args:
        matches_df: Full match DataFrame with match_date, home_team_id,
            away_team_id, venue columns.
        team_id: Team to evaluate.
        match_date: Reference date (the upcoming match). Games on or after
            this date are excluded.
        window_days: Look-back period in days.

    Returns:
        Fatigue score (float, typically 0 to ~6).
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
    recent = df[mask]

    fatigue = 0.0
    for _, row in recent.iterrows():
        is_home = row["home_team_id"] == team_id
        if is_home:
            fatigue += 1.0
        else:
            # Estimate travel distance for away game
            home_venue = row.get("venue", "")
            # Try to figure out the team's home city from their home matches
            team_home = df[df["home_team_id"] == team_id]
            team_venue = (
                team_home["venue"].mode().iloc[0]
                if "venue" in team_home.columns and not team_home["venue"].mode().empty
                else ""
            )
            dist = _estimate_distance_km(team_venue, home_venue)
            if dist > _LONG_TRAVEL_KM:
                fatigue += _TRAVEL_FATIGUE_MULTIPLIER
            else:
                fatigue += 1.0

    return round(fatigue, 2)


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

    urgency = 0.5 + 0.5 * progress
    importance = min(zone_score * urgency, 1.0)
    return round(importance, 4)


def _venue_is_high_altitude(venue: str | None) -> bool:
    """Return True when the venue string indicates a high-altitude city."""
    if not venue:
        return False
    venue_lower = venue.lower()
    return any(city in venue_lower for city in _HIGH_ALTITUDE_CITIES)


def _get_altitude_factor(venue: str | None) -> float:
    """Return the altitude factor for a venue using config ALTITUDE_FACTOR.

    Args:
        venue: Venue or city string.

    Returns:
        Altitude penalty factor (>= 1.0).
    """
    if not venue:
        return ALTITUDE_FACTOR.get("default", 1.0)
    venue_lower = venue.lower()
    for city, factor in ALTITUDE_FACTOR.items():
        if city == "default":
            continue
        if city.lower() in venue_lower:
            return factor
    return ALTITUDE_FACTOR.get("default", 1.0)


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

    home_fatigue_list: list[float] = []
    away_fatigue_list: list[float] = []
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
        altitude_list.append(_get_altitude_factor(venue))

    df["home_fatigue"] = home_fatigue_list
    df["away_fatigue"] = away_fatigue_list
    df["home_importance"] = home_importance_list
    df["away_importance"] = away_importance_list
    df["altitude_factor"] = altitude_list

    logger.info(
        "build_context_features: added context columns to %d rows.", len(df)
    )
    return df
