"""Betting-market probability derivation from Dixon-Coles parameters.

Converts Poisson rate parameters (λ_home, λ_away) and the score matrix
into all standard markets: 1X2, BTTS, Over/Under, exact scores.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from config import DECAY_RATE, MAX_GOALS
from model.dixon_coles import dixon_coles_correction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------


def compute_score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """Compute the joint probability matrix P(home=i, away=j).

    Applies the Dixon-Coles τ correction for low-scoring outcomes
    (i + j ≤ 2).

    Args:
        lambda_home: Expected home goals (Poisson rate).
        lambda_away: Expected away goals (Poisson rate).
        rho: Dixon-Coles dependence parameter (≤ 0).
        max_goals: Grid truncation — matrix is (max_goals+1)×(max_goals+1).

    Returns:
        2-D numpy array of shape (max_goals+1, max_goals+1) where element
        [i, j] = P(home scores i, away scores j).
    """
    size = max_goals + 1
    matrix = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            p_home = poisson.pmf(i, lambda_home)
            p_away = poisson.pmf(j, lambda_away)
            tau = dixon_coles_correction(i, j, lambda_home, lambda_away, rho)
            matrix[i, j] = p_home * p_away * tau

    # Normalise to correct for truncation
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


# ---------------------------------------------------------------------------
# Market functions
# ---------------------------------------------------------------------------


def compute_1x2(score_matrix: np.ndarray) -> dict[str, float]:
    """Derive 1X2 probabilities from a score matrix.

    Args:
        score_matrix: 2-D array as returned by compute_score_matrix().

    Returns:
        Dict with keys 'home', 'draw', 'away', each a probability in [0,1]
        that sums to 1.0.
    """
    n = score_matrix.shape[0]
    indices = np.arange(n)
    i, j = np.meshgrid(indices, indices, indexing="ij")

    p_home = float(score_matrix[i > j].sum())
    p_draw = float(np.trace(score_matrix))
    p_away = float(score_matrix[i < j].sum())

    total = p_home + p_draw + p_away
    if total <= 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}

    return {
        "home": p_home / total,
        "draw": p_draw / total,
        "away": p_away / total,
    }


def compute_btts(score_matrix: np.ndarray) -> float:
    """Compute the probability that both teams score (BTTS Yes).

    Args:
        score_matrix: 2-D array as returned by compute_score_matrix().

    Returns:
        P(home ≥ 1 AND away ≥ 1) as a float in [0, 1].
    """
    # P(BTTS) = 1 - P(home=0) - P(away=0) + P(home=0, away=0)
    p_btts = float(score_matrix[1:, 1:].sum())
    return p_btts


def compute_over_under(
    score_matrix: np.ndarray, line: float = 2.5
) -> dict[str, float]:
    """Derive Over/Under market probabilities.

    Args:
        score_matrix: 2-D array as returned by compute_score_matrix().
        line: Goal line (e.g. 2.5 means over requires ≥ 3 total goals).

    Returns:
        Dict with keys 'over' and 'under', summing to 1.0.
    """
    n = score_matrix.shape[0]
    p_over = 0.0
    for i in range(n):
        for j in range(n):
            if i + j > line:
                p_over += score_matrix[i, j]

    p_over = float(np.clip(p_over, 0.0, 1.0))
    return {"over": p_over, "under": 1.0 - p_over}


def compute_exact_scores(
    score_matrix: np.ndarray,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Return the top-N most probable exact scorelines.

    Args:
        score_matrix: 2-D array as returned by compute_score_matrix().
        top_n: Number of scorelines to return.

    Returns:
        List of dicts sorted by descending probability, each with keys:
        'score' (str, e.g. '1-0') and 'prob' (float).
    """
    n = score_matrix.shape[0]
    results: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            results.append(
                {"score": f"{i}-{j}", "prob": float(score_matrix[i, j])}
            )

    results.sort(key=lambda d: d["prob"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Context adjustment helpers
# ---------------------------------------------------------------------------


def _adjust_lambda(
    base_lambda: float,
    form: float,
    fatigue: int,
    importance: float,
    altitude_factor: float = 1.0,
    is_home: bool = False,
) -> float:
    """Apply contextual adjustments to a raw λ value.

    Args:
        base_lambda: Raw expected goals from the Dixon-Coles parameters.
        form: Recent form score in [0, 1].
        fatigue: Number of games played in the preceding 21 days (0–6+).
        importance: Match importance score in [0, 1].
        altitude_factor: 1.0 normally, > 1.0 penalises visiting team.
        is_home: True when adjusting the home team's lambda.

    Returns:
        Adjusted λ value (positive float).
    """
    # Form adjustment: centred at 0.5, ±20% effect
    form_adj = 1.0 + 0.4 * (form - 0.5)

    # Fatigue: each extra game beyond 2 in 21 days reduces output by 2%
    fatigue_adj = max(1.0 - 0.02 * max(fatigue - 2, 0), 0.85)

    # Importance: low-stakes teams slightly underperform (±5%)
    importance_adj = 0.95 + 0.1 * importance

    # Altitude: penalises the visiting team
    alt_adj = 1.0 / altitude_factor if not is_home else 1.0

    adjusted = base_lambda * form_adj * fatigue_adj * importance_adj * alt_adj
    return max(adjusted, 0.01)


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_match(
    home_team_id: int,
    away_team_id: int,
    posterior_means: pd.DataFrame,
    home_form: float = 0.5,
    away_form: float = 0.5,
    context_features: dict[str, Any] | None = None,
    rho: float = -0.1,
    home_advantage: float = 0.3,
    intercept: float = 0.0,
) -> dict[str, Any]:
    """Compute all betting-market probabilities for a single match.

    Integrates team strengths, form, fatigue, importance and altitude to
    produce a full probability profile.

    Args:
        home_team_id: API-Football team id for the home side.
        away_team_id: API-Football team id for the away side.
        posterior_means: DataFrame with columns team_id (or index),
            attack_mean, defense_mean.
        home_form: Recent form score for the home team in [0, 1].
        away_form: Recent form score for the away team in [0, 1].
        context_features: Optional dict with keys: home_fatigue, away_fatigue,
            home_importance, away_importance, altitude_factor.
        rho: Dixon-Coles ρ parameter.
        home_advantage: Log-scale home advantage.
        intercept: Model intercept.

    Returns:
        Dict with keys: lambda_home, lambda_away, score_matrix (list of lists),
        markets_1x2, btts, over_under, exact_scores.
    """
    ctx = context_features or {}

    def _get_param(df: pd.DataFrame, team_id: int, col: str) -> float:
        row = df[df["team_id"] == team_id] if "team_id" in df.columns else df
        if row.empty:
            logger.warning("Team %d not found in posterior_means.", team_id)
            return 0.0
        return float(row[col].iloc[0])

    attack_h = _get_param(posterior_means, home_team_id, "attack_mean")
    defense_h = _get_param(posterior_means, home_team_id, "defense_mean")
    attack_a = _get_param(posterior_means, away_team_id, "attack_mean")
    defense_a = _get_param(posterior_means, away_team_id, "defense_mean")

    raw_lambda_home = np.exp(intercept + home_advantage + attack_h + defense_a)
    raw_lambda_away = np.exp(intercept + attack_a + defense_h)

    altitude_factor = float(ctx.get("altitude_factor", 1.0))

    lambda_home = _adjust_lambda(
        raw_lambda_home,
        form=home_form,
        fatigue=int(ctx.get("home_fatigue", 0)),
        importance=float(ctx.get("home_importance", 0.5)),
        altitude_factor=1.0,
        is_home=True,
    )
    lambda_away = _adjust_lambda(
        raw_lambda_away,
        form=away_form,
        fatigue=int(ctx.get("away_fatigue", 0)),
        importance=float(ctx.get("away_importance", 0.5)),
        altitude_factor=altitude_factor,
        is_home=False,
    )

    score_mat = compute_score_matrix(lambda_home, lambda_away, rho)
    markets_1x2 = compute_1x2(score_mat)
    btts = compute_btts(score_mat)
    over_under = compute_over_under(score_mat)
    exact_scores = compute_exact_scores(score_mat)

    logger.info(
        "predict_match: home=%d λ=%.3f, away=%d λ=%.3f → 1X2 %.2f/%.2f/%.2f",
        home_team_id,
        lambda_home,
        away_team_id,
        lambda_away,
        markets_1x2["home"],
        markets_1x2["draw"],
        markets_1x2["away"],
    )

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "score_matrix": score_mat.tolist(),
        "markets_1x2": markets_1x2,
        "btts": btts,
        "over_under": over_under,
        "exact_scores": exact_scores,
    }
