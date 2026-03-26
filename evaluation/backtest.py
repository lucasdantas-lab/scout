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
    include_context: bool = False,
) -> pd.DataFrame:
    """Run an expanding-window walk-forward backtest.

    For each round from round 10 of the 3rd season onward:
      1. Trains a Dixon-Coles MLE model on all **prior** matches.
      2. Predicts each match in the current round.
      3. Records prediction vs actual result.

    When ``include_context`` is True, also generates a context-adjusted
    prediction for comparison (using form and fatigue features).

    Args:
        matches_df: DataFrame of finished matches with columns:
            id, season, round, match_date, home_team_id, away_team_id,
            home_goals, away_goals. Optionally: home_form, away_form,
            home_fatigue, away_fatigue for context mode.
        min_train_seasons: Minimum number of complete seasons that must be
            available before predictions start.
        include_context: If True, also compute context-adjusted predictions
            and add prob_home_ctx / prob_draw_ctx / prob_away_ctx / rps_ctx.

    Returns:
        DataFrame with columns: match_id, season, round, match_date,
        home_team_id, away_team_id, home_goals, away_goals,
        prob_home, prob_draw, prob_away, lambda_home, lambda_away, rps.
        When include_context is True, adds rps_ctx column.
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

            row_result: dict[str, Any] = {
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

            # Context-adjusted prediction
            if include_context:
                from model.markets import apply_context_adjustments
                lh_ctx, la_ctx, _ = apply_context_adjustments(
                    lambda_h, lambda_a,
                    fatigue_home=float(match.get("home_fatigue", 0)),
                    fatigue_away=float(match.get("away_fatigue", 0)),
                    importance_home=float(match.get("home_importance", 0.5)),
                    importance_away=float(match.get("away_importance", 0.5)),
                    altitude_factor=float(match.get("altitude_factor", 1.0)),
                )
                ctx_mat = compute_score_matrix(lh_ctx, la_ctx, rho)
                ctx_probs = compute_1x2(ctx_mat)
                ctx_pred = np.array([[ctx_probs["home"], ctx_probs["draw"], ctx_probs["away"]]])
                rps_ctx = ranked_probability_score(onehot, ctx_pred)
                row_result["rps_ctx"] = rps_ctx

            results.append(row_result)

    logger.info(
        "walk_forward_backtest: %d predictions generated.", len(results)
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rps_over_time(backtest_results: pd.DataFrame) -> go.Figure:
    """Plot RPS across rounds with naive and context-adjusted reference lines.

    Shows three lines: base model, model with context (if available),
    and naive model (1/3 each).

    Args:
        backtest_results: DataFrame returned by walk_forward_backtest().

    Returns:
        Plotly Figure.
    """
    df = backtest_results.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    df = df.sort_values("match_date").reset_index(drop=True)

    df["rps_rolling"] = df["rps"].rolling(window=10, min_periods=1).mean()

    naive_rps = ranked_probability_score(
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        np.full((3, 3), 1 / 3),
    )

    fig = go.Figure()

    # Base model
    fig.add_trace(
        go.Scatter(
            x=df["match_date"].tolist(),
            y=df["rps_rolling"].tolist(),
            mode="lines",
            line={"color": "royalblue", "width": 2},
            name="Modelo base",
        )
    )

    # Context-adjusted model (if available)
    if "rps_ctx" in df.columns:
        df["rps_ctx_rolling"] = df["rps_ctx"].rolling(window=10, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df["match_date"].tolist(),
                y=df["rps_ctx_rolling"].tolist(),
                mode="lines",
                line={"color": "green", "width": 2},
                name="Modelo com contexto",
            )
        )

    # Naive reference
    fig.add_hline(
        y=naive_rps,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Naive ({naive_rps:.3f})",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Walk-Forward RPS Over Time",
        xaxis_title="Data",
        yaxis_title="Ranked Probability Score",
        template="plotly_white",
        legend={"x": 0.01, "y": 0.99},
    )

    return fig


def plot_model_vs_market(
    backtest_results: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> go.Figure:
    """Compare cumulative RPS of the model vs market odds.

    Args:
        backtest_results: DataFrame from walk_forward_backtest() with
            match_id, prob_home, prob_draw, prob_away, home_goals, away_goals.
        odds_df: DataFrame with match_id, odds_home, odds_draw, odds_away.

    Returns:
        Plotly Figure with cumulative RPS lines.
    """
    from evaluation.metrics import compare_to_market

    comparison = compare_to_market(backtest_results, odds_df)
    if comparison.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sem dados de odds disponíveis", showarrow=False)
        return fig

    # Merge with dates for x-axis
    merged = comparison.merge(
        backtest_results[["match_id", "match_date"]],
        on="match_id",
        how="left",
    )
    merged["match_date"] = pd.to_datetime(merged["match_date"], utc=True)
    merged = merged.sort_values("match_date").reset_index(drop=True)

    merged["cum_rps_model"] = merged["rps_model"].expanding().mean()
    merged["cum_rps_market"] = merged["rps_market"].expanding().mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=merged["match_date"].tolist(),
            y=merged["cum_rps_model"].tolist(),
            mode="lines",
            name="Modelo SCOUT",
            line={"color": "royalblue", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=merged["match_date"].tolist(),
            y=merged["cum_rps_market"].tolist(),
            mode="lines",
            name="Mercado (odds)",
            line={"color": "orange", "width": 2},
        )
    )

    fig.update_layout(
        title="RPS Acumulado: Modelo vs Mercado",
        xaxis_title="Data",
        yaxis_title="RPS Médio Acumulado",
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
