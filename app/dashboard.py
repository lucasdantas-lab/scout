"""SCOUT — Streamlit dashboard.

Four pages:
    1. Previsões       — upcoming fixtures with predictions per round.
    2. Análise de Jogo — deep-dive for any home/away combination.
    3. Performance     — model evaluation metrics and calibration.
    4. Parâmetros      — posterior team strength rankings and diagnostics.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running directly:
# `streamlit run app/dashboard.py`
_project_root = str(Path(__file__).parent.parent.resolve())
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.match_card import render_match_card
from app.components.metrics_panel import render_metrics_panel
from app.components.score_matrix import render_score_matrix
from data.repository import MatchRepository, ModelRepository
from model.markets import (
    compute_btts,
    compute_over_under,
    compute_score_matrix,
    compute_1x2,
    compute_exact_scores,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SCOUT — Brasileirão Analytics",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_teams() -> dict[int, str]:
    """Load team id → name mapping from the database."""
    try:
        repo = MatchRepository()
        from supabase import create_client
        from config import SUPABASE_URL, SUPABASE_KEY
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = client.table("teams").select("id, name").execute()
        return {row["id"]: row["name"] for row in resp.data}
    except Exception as exc:
        logger.warning("Could not load teams: %s", exc)
        return {}


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    try:
        return ModelRepository().get_latest_predictions()
    except Exception as exc:
        logger.warning("Could not load predictions: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_predictions_with_results() -> pd.DataFrame:
    try:
        return ModelRepository().get_predictions_with_results()
    except Exception as exc:
        logger.warning("Could not load predictions with results: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_finished_matches() -> pd.DataFrame:
    try:
        return MatchRepository().get_finished_matches()
    except Exception as exc:
        logger.warning("Could not load finished matches: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_posterior_means() -> pd.DataFrame:
    try:
        return ModelRepository().get_latest_predictions()
    except Exception as exc:
        logger.warning("Could not load posterior means: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

page = st.sidebar.selectbox(
    "Página",
    ["Previsões", "Análise de Jogo", "Performance do Modelo", "Parâmetros"],
)

teams = load_teams()

# ===========================================================================
# PAGE 1 — Previsões
# ===========================================================================

if page == "Previsões":
    st.title("SCOUT — Previsões Brasileirão Série A")

    predictions_df = load_predictions()

    if predictions_df.empty:
        st.info("Sem previsões geradas. Clique em **Gerar Previsões** para começar.")
    else:
        # Filters
        rounds = sorted(predictions_df["round"].dropna().unique()) if "round" in predictions_df.columns else []
        selected_round = st.sidebar.selectbox("Rodada", ["Todas"] + list(rounds))

        filtered = predictions_df.copy()
        if selected_round != "Todas":
            filtered = filtered[filtered["round"] == selected_round]

        team_options = ["Todos"] + [
            teams.get(tid, str(tid))
            for tid in set(filtered.get("home_team_id", pd.Series()).tolist()
                           + filtered.get("away_team_id", pd.Series()).tolist())
        ]
        selected_team = st.sidebar.selectbox("Time", team_options)
        if selected_team != "Todos":
            tid = next(
                (k for k, v in teams.items() if v == selected_team), None
            )
            if tid is not None:
                filtered = filtered[
                    (filtered["home_team_id"] == tid)
                    | (filtered["away_team_id"] == tid)
                ]

        st.markdown(f"**{len(filtered)} partidas encontradas**")

        for _, row in filtered.iterrows():
            render_match_card(
                match_date=str(row.get("generated_at", "—")),
                home_team=teams.get(row.get("home_team_id"), "?"),
                away_team=teams.get(row.get("away_team_id"), "?"),
                prob_home=float(row.get("prob_home", 0.33)),
                prob_draw=float(row.get("prob_draw", 0.33)),
                prob_away=float(row.get("prob_away", 0.34)),
                prob_btts=float(row.get("prob_btts", 0.5)),
                prob_over25=float(row.get("prob_over25", 0.5)),
            )

    if st.button("Gerar Previsões"):
        st.info(
            "Pipeline de predição em desenvolvimento. "
            "Execute o script de treinamento e ingestão primeiro."
        )

# ===========================================================================
# PAGE 2 — Análise de Jogo
# ===========================================================================

elif page == "Análise de Jogo":
    st.title("Análise de Jogo")

    team_names = list(teams.values()) if teams else ["Time A", "Time B"]
    col_h, col_a = st.columns(2)
    with col_h:
        home_name = st.selectbox("Mandante", team_names, index=0)
    with col_a:
        away_name = st.selectbox("Visitante", team_names, index=min(1, len(team_names) - 1))

    # Fetch prediction from DB if available
    predictions_df = load_predictions()
    home_id = next((k for k, v in teams.items() if v == home_name), None)
    away_id = next((k for k, v in teams.items() if v == away_name), None)

    match_pred = pd.DataFrame()
    has_cols = "home_team_id" in predictions_df.columns and "away_team_id" in predictions_df.columns
    if not predictions_df.empty and has_cols and home_id and away_id:
        match_pred = predictions_df[
            (predictions_df["home_team_id"] == home_id)
            & (predictions_df["away_team_id"] == away_id)
        ].sort_values("generated_at", ascending=False).head(1)

    try:
        if match_pred.empty:
            st.info(
                "Sem previsão salva para este confronto. "
                "Exibindo análise com λ padrão (demonstração)."
            )
            lambda_home, lambda_away, rho = 1.4, 1.1, -0.1
            score_mat = compute_score_matrix(lambda_home, lambda_away, rho)
        else:
            row = match_pred.iloc[0]
            lambda_home = float(row.get("lambda_home", 1.4))
            lambda_away = float(row.get("lambda_away", 1.1))
            score_mat_raw = row.get("score_matrix")
            if score_mat_raw:
                import json
                score_mat = np.array(
                    json.loads(score_mat_raw)
                    if isinstance(score_mat_raw, str)
                    else score_mat_raw,
                    dtype=float,
                )
            else:
                score_mat = compute_score_matrix(lambda_home, lambda_away, -0.1)

        probs_1x2 = compute_1x2(score_mat)
        btts = compute_btts(score_mat)
        ou = compute_over_under(score_mat)
        top_scores = compute_exact_scores(score_mat)

        # --- 1X2 gauges ---
        st.subheader("Probabilidades 1X2")
        col1, col2, col3 = st.columns(3)
        for col, label, value, color in [
            (col1, f"Casa ({home_name})", probs_1x2["home"], "#2196F3"),
            (col2, "Empate", probs_1x2["draw"], "#9E9E9E"),
            (col3, f"Fora ({away_name})", probs_1x2["away"], "#F44336"),
        ]:
            with col:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=round(value * 100, 1),
                        number={"suffix": "%"},
                        title={"text": label},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                        },
                    )
                )
                fig.update_layout(height=250, margin={"t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True)

        # --- BTTS / Over ---
        col_b, col_o, col_lam = st.columns(3)
        with col_b:
            st.metric("BTTS (Ambas Marcam)", f"{btts:.0%}")
        with col_o:
            st.metric("Over 2.5", f"{ou['over']:.0%}")
        with col_lam:
            st.metric(f"λ — {home_name} / {away_name}", f"{lambda_home:.2f} / {lambda_away:.2f}")

        # --- Score matrix ---
        st.subheader("Matriz de Placares")
        fig_mat = render_score_matrix(score_mat, home_name, away_name)
        st.plotly_chart(fig_mat, use_container_width=True)

        # --- Top exact scores ---
        st.subheader("Placares Mais Prováveis")
        scores_df = pd.DataFrame(top_scores)
        if not scores_df.empty:
            fig_bar = go.Figure(
                go.Bar(
                    x=[f"{r['prob']:.1%}" for r in top_scores],
                    y=[r["score"] for r in top_scores],
                    orientation="h",
                    marker_color="royalblue",
                )
            )
            fig_bar.update_layout(
                title="Top 12 placares",
                xaxis_title="Probabilidade",
                yaxis={"autorange": "reversed"},
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as _exc:
        st.error(f"Erro ao renderizar a análise: {_exc}")
        logger.exception("Erro na página Análise de Jogo")

# ===========================================================================
# PAGE 3 — Performance do Modelo
# ===========================================================================

elif page == "Performance do Modelo":
    st.title("Performance do Modelo")

    preds_results = load_predictions_with_results()

    if preds_results.empty:
        st.info("Sem previsões com resultados disponíveis para avaliação.")
    else:
        from evaluation.metrics import compute_all_metrics
        from model.calibration import plot_reliability_diagram

        metrics_df = compute_all_metrics(preds_results)
        render_metrics_panel(metrics_df)

        # Reliability diagram
        st.subheader("Reliability Diagram")
        if "prob_home" in preds_results.columns and "home_goals" in preds_results.columns:
            y_true = (preds_results["home_goals"] > preds_results["away_goals"]).astype(int)
            y_pred = preds_results["prob_home"].dropna()
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) > 10:
                fig_rel = plot_reliability_diagram(
                    y_true.loc[common_idx].values,
                    y_pred.loc[common_idx].values,
                    title="Reliability Diagram — Vitória Mandante",
                )
                st.plotly_chart(fig_rel, use_container_width=True)

        # Walk-forward RPS curve
        st.subheader("RPS Walk-Forward")
        finished = load_finished_matches()
        if not finished.empty and len(finished) > 200:
            with st.spinner("Calculando backtest …"):
                try:
                    from evaluation.backtest import plot_rps_over_time, walk_forward_backtest
                    bt = walk_forward_backtest(finished, min_train_seasons=2)
                    fig_rps = plot_rps_over_time(bt)
                    st.plotly_chart(fig_rps, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Backtest falhou: {exc}")
        else:
            st.info("Dados insuficientes para backtest walk-forward.")

        # Last 20 predictions vs results
        st.subheader("Últimas 20 Previsões vs Resultado")
        display_cols = [
            c for c in [
                "match_id", "match_date", "prob_home", "prob_draw", "prob_away",
                "home_goals", "away_goals",
            ]
            if c in preds_results.columns
        ]
        st.dataframe(
            preds_results[display_cols].head(20),
            use_container_width=True,
        )

# ===========================================================================
# PAGE 4 — Parâmetros
# ===========================================================================

elif page == "Parâmetros":
    st.title("Parâmetros do Modelo")

    try:
        from config import SUPABASE_URL, SUPABASE_KEY
        from supabase import create_client

        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = (
            client.table("model_parameters")
            .select("*")
            .eq("parameter_type", "posterior_mean")
            .order("attack", desc=True)
            .execute()
        )
        params_df = pd.DataFrame(resp.data)
    except Exception as exc:
        logger.warning("Could not load model parameters: %s", exc)
        params_df = pd.DataFrame()

    if params_df.empty:
        st.info("Sem parâmetros disponíveis. Treine o modelo primeiro.")
    else:
        if "team_id" in params_df.columns:
            params_df["team"] = params_df["team_id"].map(teams)

        col_atk, col_def = st.columns(2)

        with col_atk:
            st.subheader("Ranking — Força de Ataque")
            atk_cols = [c for c in ["team", "attack"] if c in params_df.columns]
            st.dataframe(
                params_df.sort_values("attack", ascending=False)[atk_cols],
                use_container_width=True,
            )

        with col_def:
            st.subheader("Ranking — Força de Defesa (menor = melhor)")
            def_cols = [c for c in ["team", "defense"] if c in params_df.columns]
            st.dataframe(
                params_df.sort_values("defense", ascending=True)[def_cols],
                use_container_width=True,
            )

        # Violin plots for top 6 teams by attack
        if "attack" in params_df.columns and "team" in params_df.columns:
            st.subheader("Distribuições Posteriores — Top 6 Times (Ataque)")
            top6 = params_df.nlargest(6, "attack")["team"].tolist()

            try:
                resp_std = (
                    client.table("model_parameters")
                    .select("*")
                    .eq("parameter_type", "posterior_std")
                    .execute()
                )
                std_df = pd.DataFrame(resp_std.data)
                if not std_df.empty and "team_id" in std_df.columns:
                    std_df["team"] = std_df["team_id"].map(teams)
                    fig_violin = go.Figure()
                    for team_name in top6:
                        mean_row = params_df[params_df["team"] == team_name]
                        std_row = std_df[std_df["team"] == team_name]
                        if mean_row.empty or std_row.empty:
                            continue
                        mu = float(mean_row["attack"].iloc[0])
                        sigma = float(std_row["attack"].iloc[0])
                        samples = np.random.normal(mu, sigma, 500)
                        fig_violin.add_trace(
                            go.Violin(
                                y=samples.tolist(),
                                name=team_name,
                                box_visible=True,
                                meanline_visible=True,
                            )
                        )
                    fig_violin.update_layout(
                        title="Posterior Ataque — Top 6 Times",
                        template="plotly_white",
                        yaxis_title="Parâmetro de Ataque",
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
            except Exception as exc:
                logger.warning("Could not render violin plots: %s", exc)

        # MCMC trace plots if a trace file exists
        st.subheader("Trace Plots MCMC")
        from pathlib import Path
        trace_dir = Path(__file__).parent.parent / "traces"
        trace_files = sorted(trace_dir.glob("*.nc"), reverse=True)
        if trace_files:
            latest_trace = trace_files[0]
            st.caption(f"Arquivo de trace: `{latest_trace.name}`")
            try:
                import arviz as az
                idata = az.from_netcdf(str(latest_trace))
                axes = az.plot_trace(
                    idata,
                    var_names=["home_advantage", "rho", "intercept"],
                    show=False,
                )
                import matplotlib.pyplot as plt
                fig_trace = plt.gcf()
                st.pyplot(fig_trace)
                plt.close()
            except Exception as exc:
                st.warning(f"Não foi possível carregar o trace: {exc}")
        else:
            st.info("Nenhum trace MCMC encontrado em `traces/`.")
