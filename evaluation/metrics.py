"""Model evaluation metrics.

Implements Brier Score, Ranked Probability Score, Log-Loss, and comparison
helpers for football prediction quality assessment.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def brier_score(
    y_true_onehot: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Compute the multi-class Brier Score for 1X2 predictions.

    Args:
        y_true_onehot: Array of shape (N, 3) — one-hot encoded outcomes
            in order [home_win, draw, away_win].
        y_pred_proba: Array of shape (N, 3) — predicted probabilities.

    Returns:
        Mean Brier Score (lower is better; 0 = perfect, 2 = worst).
    """
    y_true_onehot = np.asarray(y_true_onehot, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    if y_true_onehot.shape != y_pred_proba.shape:
        raise ValueError(
            "y_true_onehot and y_pred_proba must have the same shape. "
            f"Got {y_true_onehot.shape} vs {y_pred_proba.shape}."
        )
    return float(np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1)))


def ranked_probability_score(
    y_true_onehot: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Compute the Ranked Probability Score (RPS) for ordinal outcomes.

    For football results the ordering is: away_win < draw < home_win.
    This function assumes the column order is [home_win, draw, away_win].

    Args:
        y_true_onehot: Array of shape (N, 3) — one-hot encoded outcomes.
        y_pred_proba: Array of shape (N, 3) — predicted probabilities.

    Returns:
        Mean RPS (lower is better).
    """
    y_true = np.asarray(y_true_onehot, dtype=float)
    y_pred = np.asarray(y_pred_proba, dtype=float)

    # Cumulative distributions
    cum_pred = np.cumsum(y_pred, axis=1)[:, :-1]   # shape (N, 2)
    cum_true = np.cumsum(y_true, axis=1)[:, :-1]   # shape (N, 2)

    rps_per_match = np.mean((cum_pred - cum_true) ** 2, axis=1)
    return float(np.mean(rps_per_match))


def log_loss_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Compute multi-class log-loss.

    Args:
        y_true: Integer array of shape (N,) with class labels 0/1/2
            (0=home win, 1=draw, 2=away win).
        y_pred_proba: Array of shape (N, 3) — predicted probabilities.

    Returns:
        Mean log-loss (lower is better).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.clip(np.asarray(y_pred_proba, dtype=float), 1e-12, 1.0)
    n = len(y_true)
    log_likelihoods = np.log(y_pred[np.arange(n), y_true])
    return float(-np.mean(log_likelihoods))


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def _encode_result(home_goals: int, away_goals: int) -> int:
    """Return class label: 0=home win, 1=draw, 2=away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def compute_all_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all evaluation metrics on a predictions DataFrame.

    Args:
        predictions_df: Must contain columns: prob_home, prob_draw, prob_away,
            home_goals, away_goals (the actual results).

    Returns:
        Single-row DataFrame with columns: brier_score, rps, log_loss,
        n_samples.
    """
    required = {"prob_home", "prob_draw", "prob_away", "home_goals", "away_goals"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df is missing columns: {missing}")

    df = predictions_df.dropna(
        subset=["prob_home", "prob_draw", "prob_away", "home_goals", "away_goals"]
    )
    if df.empty:
        logger.warning("compute_all_metrics: no complete rows to evaluate.")
        return pd.DataFrame(
            [{"brier_score": None, "rps": None, "log_loss": None, "n_samples": 0}]
        )

    y_pred = df[["prob_home", "prob_draw", "prob_away"]].values
    y_labels = [
        _encode_result(int(r.home_goals), int(r.away_goals))
        for _, r in df.iterrows()
    ]
    y_onehot = np.eye(3, dtype=float)[y_labels]

    bs = brier_score(y_onehot, y_pred)
    rps = ranked_probability_score(y_onehot, y_pred)
    ll = log_loss_score(np.array(y_labels), y_pred)

    logger.info(
        "Metrics on %d samples: BS=%.4f, RPS=%.4f, LL=%.4f",
        len(df), bs, rps, ll,
    )

    return pd.DataFrame(
        [{"brier_score": bs, "rps": rps, "log_loss": ll, "n_samples": len(df)}]
    )


def compare_to_market(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare model RPS to the RPS implied by market odds.

    Args:
        predictions_df: Must contain columns: match_id, prob_home, prob_draw,
            prob_away, home_goals, away_goals.
        odds_df: Must contain columns: match_id, odds_home, odds_draw,
            odds_away (decimal odds).

    Returns:
        DataFrame with columns: match_id, rps_model, rps_market, rps_delta
        (model – market; negative means model is better).
    """
    merged = predictions_df.merge(odds_df, on="match_id", how="inner")
    if merged.empty:
        logger.warning("compare_to_market: no matching match_ids.")
        return pd.DataFrame(
            columns=["match_id", "rps_model", "rps_market", "rps_delta"]
        )

    records = []
    for _, row in merged.iterrows():
        if any(
            pd.isna(row.get(c))
            for c in ["prob_home", "prob_draw", "prob_away",
                       "odds_home", "odds_draw", "odds_away",
                       "home_goals", "away_goals"]
        ):
            continue

        true_label = _encode_result(int(row.home_goals), int(row.away_goals))
        onehot = np.eye(3, dtype=float)[[true_label]]

        model_probs = np.array(
            [[row.prob_home, row.prob_draw, row.prob_away]]
        )

        # Convert decimal odds to implied probability (with over-round)
        raw = np.array(
            [1 / row.odds_home, 1 / row.odds_draw, 1 / row.odds_away]
        )
        market_probs = (raw / raw.sum()).reshape(1, -1)

        rps_m = ranked_probability_score(onehot, model_probs)
        rps_mkt = ranked_probability_score(onehot, market_probs)

        records.append(
            {
                "match_id": row.match_id,
                "rps_model": rps_m,
                "rps_market": rps_mkt,
                "rps_delta": rps_m - rps_mkt,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        logger.info(
            "compare_to_market: mean RPS delta = %.4f over %d matches.",
            result["rps_delta"].mean(),
            len(result),
        )
    return result
