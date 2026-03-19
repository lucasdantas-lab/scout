"""Walk-forward backtesting for the SCOUT prediction model.

Validates the model via an expanding-window scheme that strictly avoids
look-ahead bias: the model is always trained exclusively on matches that
occurred before the fixture being predicted.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from evaluation.metrics import ranked_probability_score
from model.dixon_coles import fit_dixon_coles_mle
from model.markets import compute_score_matrix, compute_1x2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


def walk_forward_backtest(
    matches_df: pd.DataFrame,
    min_train_seasons: int = 2,
) -> pd.DataFrame:
    """Run an expanding-window walk-forward backtest.

    For each round from round 10 of the 3rd season onward:
      1. Trains a Dixon-Coles MLE model on all **prior** matches.
      2. Predicts each match in the current round.
      3. Records prediction vs actual result.

    Args:
        matches_df: DataFrame of finished matches with columns:
            id, season, round, match_date, home_team_id, away_team_id,
            home_goals, away_goals.
        min_train_seasons: Minimum number of complete seasons that must be
            available before predictions start.

    Returns:
        DataFrame with columns: match_id, season, round, match_date,
        home_team_id, away_team_id, home_goals, away_goals,
        prob_home, prob_draw, prob_away, lambda_home, lambda_away, rps.
    """
    df = matches_df.dropna(
        subset=[
            "home_goals", "away_goals", "home_team_id", "away_team_id",
            "match_date", "season", "round",
        ]
    ).copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    df = df.sort_values("match_date").reset_index(drop=True)

    # Extract numeric round
    df["round_num"] = df["round"].apply(_parse_round)

    seasons_sorted = sorted(df["season"].unique())
    if len(seasons_sorted) < min_train_seasons:
        raise ValueError(
            f"Need at least {min_train_seasons} seasons of data, "
            f"got {len(seasons_sorted)}."
        )

    start_season = seasons_sorted[min_train_seasons - 1]
    cutoff_round = 10

    results: list[dict[str, Any]] = []

    # Unique (season, round) combinations from the start point onwards
    predict_rounds = (
        df[
            (df["season"] > start_season)
            | ((df["season"] == start_season) & (df["round_num"] >= cutoff_round))
        ]
        .groupby(["season", "round_num"])
        .groups
    )

    for (season, round_num) in sorted(predict_rounds.keys()):
        predict_mask = (df["season"] == season) & (df["round_num"] == round_num)
        predict_matches = df[predict_mask]

        # Training data: strictly before this round
        train_cutoff = predict_matches["match_date"].min()
        train_df = df[df["match_date"] < train_cutoff]

        if len(train_df) < 50:
            logger.debug(
                "Skipping season=%d round=%d — insufficient training data (%d rows).",
                season, round_num, len(train_df),
            )
            continue

        try:
            fit = fit_dixon_coles_mle(train_df)
        except Exception as exc:
            logger.warning(
                "MLE fit failed for season=%d round=%d: %s", season, round_num, exc
            )
            continue

        team_params = fit["team_params"]
        home_adv = fit["home_advantage"]
        rho = fit["rho"]

        for _, match in predict_matches.iterrows():
            h_id = match["home_team_id"]
            a_id = match["away_team_id"]

            if h_id not in team_params or a_id not in team_params:
                continue

            lambda_h = np.exp(
                home_adv
                + team_params[h_id]["attack"]
                + team_params[a_id]["defense"]
            )
            lambda_a = np.exp(
                team_params[a_id]["attack"]
                + team_params[h_id]["defense"]
            )

            score_mat = compute_score_matrix(lambda_h, lambda_a, rho)
            probs = compute_1x2(score_mat)

            true_label = _result_label(int(match["home_goals"]), int(match["away_goals"]))
            onehot = np.eye(3, dtype=float)[[true_label]]
            pred_array = np.array([[probs["home"], probs["draw"], probs["away"]]])
            rps_val = ranked_probability_score(onehot, pred_array)

            results.append(
                {
                    "match_id": match.get("id"),
                    "season": season,
                    "round": match["round"],
                    "match_date": match["match_date"],
                    "home_team_id": h_id,
                    "away_team_id": a_id,
                    "home_goals": int(match["home_goals"]),
                    "away_goals": int(match["away_goals"]),
                    "prob_home": probs["home"],
                    "prob_draw": probs["draw"],
                    "prob_away": probs["away"],
                    "lambda_home": lambda_h,
                    "lambda_away": lambda_a,
                    "rps": rps_val,
                }
            )

    logger.info(
        "walk_forward_backtest: %d predictions generated.", len(results)
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rps_over_time(backtest_results: pd.DataFrame) -> go.Figure:
    """Plot RPS across rounds with a naive-model reference line.

    The naive model assigns equal probabilities (1/3) to each outcome,
    yielding an RPS of 1/3 ≈ 0.222.

    Args:
        backtest_results: DataFrame returned by walk_forward_backtest().

    Returns:
        Plotly Figure.
    """
    df = backtest_results.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)

    # Rolling 10-match mean for smoothing
    df = df.sort_values("match_date").reset_index(drop=True)
    df["rps_rolling"] = df["rps"].rolling(window=10, min_periods=1).mean()

    naive_rps = ranked_probability_score(
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),   # dummy
        np.full((3, 3), 1 / 3),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["match_date"].tolist(),
            y=df["rps"].tolist(),
            mode="markers",
            marker={"color": "lightblue", "size": 4, "opacity": 0.6},
            name="RPS (per match)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["match_date"].tolist(),
            y=df["rps_rolling"].tolist(),
            mode="lines",
            line={"color": "royalblue", "width": 2},
            name="RPS (10-match rolling avg)",
        )
    )

    fig.add_hline(
        y=naive_rps,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Naive model ({naive_rps:.3f})",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Walk-Forward RPS Over Time",
        xaxis_title="Match date",
        yaxis_title="Ranked Probability Score",
        template="plotly_white",
        legend={"x": 0.01, "y": 0.99},
    )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_round(round_str: Any) -> int:
    """Extract the numeric part from a round string like 'Regular Season - 5'."""
    try:
        return int(str(round_str).split("-")[-1].strip())
    except (ValueError, IndexError):
        return 0


def _result_label(home_goals: int, away_goals: int) -> int:
    """Return 0=home win, 1=draw, 2=away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2
