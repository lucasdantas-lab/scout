"""Shared pytest fixtures for SCOUT tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_matches() -> pd.DataFrame:
    """Synthetic finished-match DataFrame for unit testing."""
    rng = np.random.default_rng(42)
    n = 120  # 6 teams × 20 home games each
    teams = list(range(1, 7))

    rows = []
    for i in range(n):
        home = rng.choice(teams)
        away = rng.choice([t for t in teams if t != home])
        rows.append(
            {
                "id": i + 1,
                "season": 2023,
                "round": f"Regular Season - {(i // 5) + 1}",
                "match_date": pd.Timestamp("2023-01-01")
                + pd.Timedelta(days=int(i * 3.5)),
                "home_team_id": home,
                "away_team_id": away,
                "home_goals": int(rng.poisson(1.5)),
                "away_goals": int(rng.poisson(1.1)),
                "status": "FT",
                "venue": "Estádio Sintético",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def two_team_matches() -> pd.DataFrame:
    """Minimal two-team DataFrame for Dixon-Coles tests."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "season": 2023,
                "round": "Regular Season - 1",
                "match_date": pd.Timestamp("2023-03-01", tz="UTC"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_goals": 2,
                "away_goals": 1,
                "status": "FT",
                "venue": "Estádio A",
            },
            {
                "id": 2,
                "season": 2023,
                "round": "Regular Season - 2",
                "match_date": pd.Timestamp("2023-03-15", tz="UTC"),
                "home_team_id": 2,
                "away_team_id": 1,
                "home_goals": 0,
                "away_goals": 0,
                "status": "FT",
                "venue": "Estádio B",
            },
        ]
    )
