"""Dixon-Coles model fitted via Maximum Likelihood Estimation.

Reference: Dixon, M. J., & Coles, S. G. (1997). Modelling association
football scores and inefficiencies in the football betting market.
Applied Statistics, 46(2), 265–280.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from config import DECAY_RATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dixon-Coles tau correction
# ---------------------------------------------------------------------------


def dixon_coles_correction(
    x: int,
    y: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Compute the Dixon-Coles τ (tau) correction factor.

    Adjusts the joint Poisson probabilities for scores where the total
    number of goals is at most 2, correcting for the known statistical
    dependence between home and away goals in those cases.

    Args:
        x: Home goals scored.
        y: Away goals scored.
        lambda_home: Expected home goals (Poisson rate).
        lambda_away: Expected away goals (Poisson rate).
        rho: Dependence parameter (typically negative, ≤ 0).

    Returns:
        Multiplicative correction factor τ(x, y).
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if x == 1 and y == 0:
        return 1.0 + lambda_away * rho
    if x == 0 and y == 1:
        return 1.0 + lambda_home * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------


def dixon_coles_log_likelihood(
    params: np.ndarray,
    matches_df: pd.DataFrame,
    team_list: list[int],
    decay_rate: float = DECAY_RATE,
    use_xg: bool = False,
) -> float:
    """Compute the (negative) Dixon-Coles log-likelihood.

    Intended to be minimised by scipy.optimize.minimize.

    Param layout (for N teams):
        params[0 : N]        → attack parameters
        params[N : 2*N]      → defence parameters
        params[2*N]          → home_advantage
        params[2*N + 1]      → rho

    Args:
        params: Flat parameter vector.
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date,
            and optionally home_xg, away_xg.
        team_list: Ordered list of team ids matching parameter indices.
        decay_rate: Exponential decay applied as decay^(days_ago / 30).
        use_xg: If True, use xG as the response variable instead of goals
            (falls back to goals when xG is not available for a match).

    Returns:
        Negative log-likelihood (scalar, to be minimised).
    """
    n_teams = len(team_list)
    team_idx = {tid: i for i, tid in enumerate(team_list)}

    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    df = matches_df.dropna(
        subset=["home_goals", "away_goals", "home_team_id", "away_team_id", "match_date"]
    ).copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    now = datetime.now(tz=timezone.utc)

    log_lik = 0.0
    for _, row in df.iterrows():
        hi = team_idx.get(row["home_team_id"])
        ai = team_idx.get(row["away_team_id"])
        if hi is None or ai is None:
            continue

        days_ago = max((now - row["match_date"]).days, 0)
        weight = decay_rate ** (days_ago / 30.0)

        lambda_h = np.exp(home_adv + attack[hi] + defense[ai])
        lambda_a = np.exp(attack[ai] + defense[hi])

        # Choose response variable: xG (rounded) or actual goals
        if use_xg and pd.notna(row.get("home_xg")) and pd.notna(row.get("away_xg")):
            x = int(round(float(row["home_xg"])))
            y = int(round(float(row["away_xg"])))
        else:
            x = int(row["home_goals"])
            y = int(row["away_goals"])

        tau = dixon_coles_correction(x, y, lambda_h, lambda_a, rho)
        if tau <= 0:
            continue

        ll = (
            poisson.logpmf(x, lambda_h)
            + poisson.logpmf(y, lambda_a)
            + np.log(tau)
        )
        log_lik += weight * ll

    return -log_lik


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------


def fit_dixon_coles_mle(
    matches_df: pd.DataFrame,
    decay_rate: float = DECAY_RATE,
    use_xg: bool = False,
) -> dict[str, Any]:
    """Fit the Dixon-Coles model via Maximum Likelihood Estimation.

    Uses L-BFGS-B with an identification constraint: the mean of the
    attack parameters equals zero.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date,
            and optionally home_xg, away_xg.
        decay_rate: Temporal decay rate.
        use_xg: If True, use xG as response variable (with goals fallback).

    Returns:
        Dict with keys:
            - 'team_params': dict mapping team_id →
                {'attack': float, 'defense': float}
            - 'home_advantage': float
            - 'rho': float
            - 'success': bool
            - 'message': str (from scipy optimiser)

    Raises:
        ValueError: When matches_df contains fewer than 2 distinct teams.
    """
    df = matches_df.dropna(
        subset=["home_goals", "away_goals", "home_team_id", "away_team_id"]
    )
    team_list = sorted(
        set(df["home_team_id"]).union(df["away_team_id"])
    )
    n_teams = len(team_list)
    if n_teams < 2:
        raise ValueError(
            f"Need at least 2 teams to fit the model, got {n_teams}."
        )

    # Initial parameter vector: zeros + small home advantage + rho=-0.1
    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 0.3      # home_advantage
    x0[2 * n_teams + 1] = -0.1  # rho

    # Bounds: rho in (-0.99, 0), everything else free
    bounds = (
        [(None, None)] * (2 * n_teams)   # attack + defense
        + [(None, None)]                   # home_advantage
        + [(-0.99, 0.0)]                   # rho
    )

    # Identification constraint: mean(attack) = 0
    constraints = [
        {
            "type": "eq",
            "fun": lambda p: p[:n_teams].mean(),
        }
    ]

    logger.info(
        "Fitting Dixon-Coles MLE for %d teams on %d matches …",
        n_teams,
        len(df),
    )

    result = minimize(
        dixon_coles_log_likelihood,
        x0,
        args=(df, team_list, decay_rate, use_xg),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        logger.warning("MLE optimisation did not converge: %s", result.message)

    params = result.x
    attack_params = params[:n_teams]
    defense_params = params[n_teams : 2 * n_teams]
    home_advantage = float(params[2 * n_teams])
    rho = float(params[2 * n_teams + 1])

    team_params = {
        tid: {
            "attack": float(attack_params[i]),
            "defense": float(defense_params[i]),
        }
        for i, tid in enumerate(team_list)
    }

    logger.info(
        "MLE fit complete. home_advantage=%.4f, rho=%.4f", home_advantage, rho
    )

    return {
        "team_params": team_params,
        "home_advantage": home_advantage,
        "rho": rho,
        "success": result.success,
        "message": result.message,
    }
