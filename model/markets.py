"""Betting-market probability derivation from Dixon-Coles parameters.

Converts Poisson rate parameters (λ_home, λ_away) and the score matrix
into all standard markets: 1X2, BTTS, Over/Under, exact scores.
Includes full contextual adjustment pipeline integrating agent-derived
desfalque data, squad strength, fatigue, and importance.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from config import MAX_GOALS
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


def compute_btts(
    lambda_home: float | None = None,
    lambda_away: float | None = None,
    score_matrix: np.ndarray | None = None,
) -> float:
    """Compute the probability that both teams score (BTTS Yes).

    Can use either direct Poisson calculation from lambdas or the score
    matrix. If lambdas are provided they take priority.

    Args:
        lambda_home: Expected home goals.
        lambda_away: Expected away goals.
        score_matrix: 2-D array as returned by compute_score_matrix().

    Returns:
        P(home ≥ 1 AND away ≥ 1) as a float in [0, 1].
    """
    if lambda_home is not None and lambda_away is not None:
        return (1.0 - np.exp(-lambda_home)) * (1.0 - np.exp(-lambda_away))
    if score_matrix is not None:
        return float(score_matrix[1:, 1:].sum())
    return 0.5


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
# Context adjustment
# ---------------------------------------------------------------------------


def apply_context_adjustments(
    lambda_home: float,
    lambda_away: float,
    context_json: dict[str, Any] | None = None,
    squad_home: float = 1.0,
    squad_away: float = 1.0,
    fatigue_home: float = 0.0,
    fatigue_away: float = 0.0,
    importance_home: float = 0.5,
    importance_away: float = 0.5,
    altitude_factor: float = 1.0,
) -> tuple[float, float, list[str]]:
    """Apply all contextual adjustments to raw lambda values.

    Adjusts λ based on:
    - ``lambda_delta`` from the context agent (desfalques)
    - squad_strength normalised
    - fatigue (penalises tired team)
    - importance (amplifies motivated team)
    - altitude (penalises low-altitude visitor)

    Args:
        lambda_home: Raw expected home goals.
        lambda_away: Raw expected away goals.
        context_json: Processed context from context_agent with 'home'
            and 'away' keys containing 'lambda_delta' and 'confianca'.
        squad_home: Home squad strength (1.0 = full strength).
        squad_away: Away squad strength.
        fatigue_home: Home fatigue score.
        fatigue_away: Away fatigue score.
        importance_home: Home importance score [0, 1].
        importance_away: Away importance score [0, 1].
        altitude_factor: Altitude penalty for away team (>= 1.0).

    Returns:
        Tuple of (lambda_home_adj, lambda_away_adj, adjustment_log).
    """
    log: list[str] = []

    lh = lambda_home
    la = lambda_away

    # 1. Context agent lambda_delta (desfalques)
    if context_json:
        home_ctx = context_json.get("home", {})
        away_ctx = context_json.get("away", {})

        h_delta = float(home_ctx.get("lambda_delta", 0.0))
        h_conf = float(home_ctx.get("confianca", 0.5))
        a_delta = float(away_ctx.get("lambda_delta", 0.0))
        a_conf = float(away_ctx.get("confianca", 0.5))

        lh *= (1.0 + h_delta * h_conf)
        la *= (1.0 + a_delta * a_conf)

        if h_delta != 0:
            log.append(f"context_home: Δ={h_delta:+.3f} conf={h_conf:.2f}")
        if a_delta != 0:
            log.append(f"context_away: Δ={a_delta:+.3f} conf={a_conf:.2f}")

    # 2. Squad strength
    if squad_home != 1.0:
        lh *= squad_home
        log.append(f"squad_home: {squad_home:.3f}")
    if squad_away != 1.0:
        la *= squad_away
        log.append(f"squad_away: {squad_away:.3f}")

    # 3. Fatigue: each game beyond 2 in 21 days reduces λ by 2%
    if fatigue_home > 2:
        fatigue_adj_h = max(1.0 - 0.02 * (fatigue_home - 2), 0.85)
        lh *= fatigue_adj_h
        log.append(f"fatigue_home: {fatigue_adj_h:.3f} (games={fatigue_home:.1f})")
    if fatigue_away > 2:
        fatigue_adj_a = max(1.0 - 0.02 * (fatigue_away - 2), 0.85)
        la *= fatigue_adj_a
        log.append(f"fatigue_away: {fatigue_adj_a:.3f} (games={fatigue_away:.1f})")

    # 4. Importance: motivated teams get slight boost (±5%)
    imp_adj_h = 0.95 + 0.1 * importance_home
    imp_adj_a = 0.95 + 0.1 * importance_away
    lh *= imp_adj_h
    la *= imp_adj_a
    if importance_home != 0.5:
        log.append(f"importance_home: {imp_adj_h:.3f}")
    if importance_away != 0.5:
        log.append(f"importance_away: {imp_adj_a:.3f}")

    # 5. Altitude: penalises the visiting team
    if altitude_factor > 1.0:
        la /= altitude_factor
        log.append(f"altitude: away λ / {altitude_factor:.2f}")

    lh = max(lh, 0.01)
    la = max(la, 0.01)

    return lh, la, log


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_match(
    home_team_id: int,
    away_team_id: int,
    posterior_means: pd.DataFrame,
    features_dict: dict[str, Any] | None = None,
    context_json: dict[str, Any] | None = None,
    rho: float = -0.1,
    home_advantage: float = 0.3,
    intercept: float = 0.0,
) -> dict[str, Any]:
    """Compute all betting-market probabilities for a single match.

    Integrates team strengths, form, fatigue, importance, altitude,
    squad strength, and context-agent adjustments.

    Args:
        home_team_id: API-Football team id for the home side.
        away_team_id: API-Football team id for the away side.
        posterior_means: DataFrame with columns team_id (or index),
            attack_mean, defense_mean.
        features_dict: Optional dict with keys: home_form, away_form,
            home_fatigue, away_fatigue, home_importance, away_importance,
            altitude_factor, home_squad, away_squad.
        context_json: Processed context from context_agent.
        rho: Dixon-Coles ρ parameter.
        home_advantage: Log-scale home advantage.
        intercept: Model intercept.

    Returns:
        Dict with keys: lambda_home, lambda_away, lambda_home_adjusted,
        lambda_away_adjusted, score_matrix, markets_1x2, btts,
        over_under, exact_scores, adjustment_log.
    """
    feat = features_dict or {}

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

    # Apply all contextual adjustments
    lambda_home, lambda_away, adj_log = apply_context_adjustments(
        lambda_home=raw_lambda_home,
        lambda_away=raw_lambda_away,
        context_json=context_json,
        squad_home=float(feat.get("home_squad", 1.0)),
        squad_away=float(feat.get("away_squad", 1.0)),
        fatigue_home=float(feat.get("home_fatigue", 0)),
        fatigue_away=float(feat.get("away_fatigue", 0)),
        importance_home=float(feat.get("home_importance", 0.5)),
        importance_away=float(feat.get("away_importance", 0.5)),
        altitude_factor=float(feat.get("altitude_factor", 1.0)),
    )

    score_mat = compute_score_matrix(lambda_home, lambda_away, rho)
    markets_1x2 = compute_1x2(score_mat)
    btts_prob = compute_btts(lambda_home=lambda_home, lambda_away=lambda_away)
    over_under = compute_over_under(score_mat)
    exact_scores = compute_exact_scores(score_mat)

    logger.info(
        "predict_match: home=%d λ=%.3f→%.3f, away=%d λ=%.3f→%.3f "
        "→ 1X2 %.2f/%.2f/%.2f",
        home_team_id, raw_lambda_home, lambda_home,
        away_team_id, raw_lambda_away, lambda_away,
        markets_1x2["home"], markets_1x2["draw"], markets_1x2["away"],
    )

    return {
        "lambda_home": raw_lambda_home,
        "lambda_away": raw_lambda_away,
        "lambda_home_adjusted": lambda_home,
        "lambda_away_adjusted": lambda_away,
        "score_matrix": score_mat.tolist(),
        "markets_1x2": markets_1x2,
        "btts": btts_prob,
        "over_under": over_under,
        "exact_scores": exact_scores,
        "adjustment_log": adj_log,
    }
