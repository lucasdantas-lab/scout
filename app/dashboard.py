"""SCOUT — Streamlit dashboard.

Five pages:
    1. Rodada            — upcoming fixtures with predictions per round.
    2. Analise de Jogo   — deep-dive for any home/away combination.
    3. Performance       — model evaluation metrics and calibration.
    4. Parametros        — posterior team strength rankings and diagnostics.
    5. Agentes           — calibration agent history and context impact.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
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
from app.components.narrative_panel import render_narrative_panel
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


def _clean_round(raw: str | None) -> str:
    """Convert 'Regular Season - 38' to 'Rodada 38'."""
    if not raw:
        return "—"
    if "Regular Season - " in str(raw):
        return "Rodada " + str(raw).split("Regular Season - ")[-1]
    return str(raw)


def _round_sort_key(r: str) -> int:
    """Extract numeric part from round string for sorting."""
    import re
    m = re.search(r"\d+", str(r))
    return int(m.group()) if m else 0


st.set_page_config(
    page_title="SCOUT — Brasileirao Analytics",
    page_icon="S",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_teams() -> dict[int, str]:
    """Load team id -> name mapping from the database."""
    try:
        from config import SUPABASE_URL, SUPABASE_KEY
        from supabase import create_client
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
def load_calibration_history() -> pd.DataFrame:
    try:
        return ModelRepository().get_calibration_history()
    except Exception as exc:
        logger.warning("Could not load calibration history: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

page = st.sidebar.selectbox(
    "Pagina",
    ["Rodada", "Analise de Jogo", "Performance do Modelo", "Parametros", "Agentes"],
)

teams = load_teams()

# ===========================================================================
# PAGE 1 — Rodada
# ===========================================================================

if page == "Rodada":
    st.title("SCOUT — Previsoes Brasileirao Serie A")

    predictions_df = load_predictions()

    if predictions_df.empty:
        st.info("Sem previsoes no banco de dados. Execute `scripts/predict.py` para gerar.")
    else:
        if "round" in predictions_df.columns:
            predictions_df["round_label"] = predictions_df["round"].apply(_clean_round)
        else:
            predictions_df["round_label"] = "—"

        # --- Sidebar filters ---
        seasons_available = sorted(
            predictions_df["season"].dropna().unique().astype(int), reverse=True
        ) if "season" in predictions_df.columns else []
        selected_season = st.sidebar.selectbox(
            "Temporada", seasons_available, index=0,
        )

        season_df = predictions_df[predictions_df["season"] == selected_season].copy()
        round_labels = sorted(
            season_df["round_label"].dropna().unique(),
            key=_round_sort_key,
            reverse=True,
        )
        selected_round = st.sidebar.selectbox("Rodada", ["Todas"] + list(round_labels))

        filtered = season_df.copy()
        if selected_round != "Todas":
            filtered = filtered[filtered["round_label"] == selected_round]

        team_ids_in_view = set(
            filtered["home_team_id"].dropna().astype(int).tolist()
            + filtered["away_team_id"].dropna().astype(int).tolist()
        ) if "home_team_id" in filtered.columns else set()
        team_options = ["Todos"] + sorted(
            [teams.get(tid, str(tid)) for tid in team_ids_in_view]
        )
        selected_team = st.sidebar.selectbox("Time", team_options)
        if selected_team != "Todos":
            tid = next((k for k, v in teams.items() if v == selected_team), None)
            if tid is not None:
                filtered = filtered[
                    (filtered["home_team_id"] == tid) | (filtered["away_team_id"] == tid)
                ]

        if "match_date" in filtered.columns:
            filtered = filtered.sort_values("match_date", ascending=False)

        st.markdown(f"**{len(filtered)} partidas** — Temporada {selected_season}")

        # Round summary button
        if selected_round != "Todas" and not filtered.empty:
            if st.button("Resumo da Rodada"):
                try:
                    from agents.narrative_agent import generate_round_summary
                    preds_list = []
                    for _, r in filtered.iterrows():
                        preds_list.append({
                            "match_label": f"{teams.get(r.get('home_team_id'), '?')} x "
                                           f"{teams.get(r.get('away_team_id'), '?')}",
                            "markets_1x2": {
                                "home": float(r.get("prob_home", 0.33)),
                                "draw": float(r.get("prob_draw", 0.33)),
                                "away": float(r.get("prob_away", 0.34)),
                            },
                            "lambda_home_adjusted": float(r.get("lambda_home_adjusted", r.get("lambda_home", 1.3))),
                            "lambda_away_adjusted": float(r.get("lambda_away_adjusted", r.get("lambda_away", 1.1))),
                            "btts": float(r.get("prob_btts", 0.5)),
                        })
                    summary = generate_round_summary(preds_list, selected_round, selected_season)
                    st.markdown(summary)
                except Exception as exc:
                    st.warning(f"Erro ao gerar resumo: {exc}")

        for _, row in filtered.iterrows():
            hg = row.get("home_goals")
            ag = row.get("away_goals")
            actual = f"{int(hg)} x {int(ag)}" if pd.notna(hg) and pd.notna(ag) else None

            raw_date = row.get("match_date", "")
            try:
                display_date = pd.to_datetime(raw_date).strftime("%-d/%m/%Y")
            except Exception:
                display_date = str(raw_date)[:10]

            # Extract context absences if available
            context_abs = []
            ctx_conf = None
            narrative = row.get("narrative")
            if isinstance(narrative, str) and narrative:
                ctx_conf = 0.5  # narrative exists = some context

            render_match_card(
                match_date=f"{display_date}  ·  {row.get('round_label', '—')}",
                home_team=teams.get(row.get("home_team_id"), "?"),
                away_team=teams.get(row.get("away_team_id"), "?"),
                prob_home=float(row.get("prob_home", 0.33)),
                prob_draw=float(row.get("prob_draw", 0.33)),
                prob_away=float(row.get("prob_away", 0.34)),
                prob_btts=float(row.get("prob_btts", 0.5)),
                prob_over25=float(row.get("prob_over25", 0.5)),
                actual_result=actual,
                context_confidence=ctx_conf,
            )

# ===========================================================================
# PAGE 2 — Analise de Jogo
# ===========================================================================

elif page == "Analise de Jogo":
    st.title("Analise de Jogo")

    team_names = list(teams.values()) if teams else ["Time A", "Time B"]
    col_h, col_a = st.columns(2)
    with col_h:
        home_name = st.selectbox("Mandante", team_names, index=0)
    with col_a:
        away_name = st.selectbox("Visitante", team_names, index=min(1, len(team_names) - 1))

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
                "Sem previsao salva para este confronto. "
                "Exibindo analise com lambda padrao (demonstracao)."
            )
            lambda_home, lambda_away, rho = 1.4, 1.1, -0.1
            lambda_home_adj, lambda_away_adj = lambda_home, lambda_away
            score_mat = compute_score_matrix(lambda_home, lambda_away, rho)
            narrative_text = None
            adj_log = []
        else:
            row = match_pred.iloc[0]
            lambda_home = float(row.get("lambda_home", 1.4))
            lambda_away = float(row.get("lambda_away", 1.1))
            lambda_home_adj = float(row.get("lambda_home_adjusted", lambda_home))
            lambda_away_adj = float(row.get("lambda_away_adjusted", lambda_away))
            narrative_text = row.get("narrative")
            adj_log = []

            score_mat_raw = row.get("score_matrix")
            if score_mat_raw:
                score_mat = np.array(
                    json.loads(score_mat_raw) if isinstance(score_mat_raw, str) else score_mat_raw,
                    dtype=float,
                )
            else:
                score_mat = compute_score_matrix(lambda_home_adj, lambda_away_adj, -0.1)

        probs_1x2 = compute_1x2(score_mat)
        btts_prob = compute_btts(lambda_home=lambda_home_adj, lambda_away=lambda_away_adj)
        ou = compute_over_under(score_mat)
        top_scores = compute_exact_scores(score_mat)

        # --- Upper section: 1X2 gauges ---
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
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
                    )
                )
                fig.update_layout(height=250, margin={"t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True)

        # --- BTTS / Over ---
        col_b, col_o, col_lam = st.columns(3)
        with col_b:
            st.metric("BTTS (Ambas Marcam)", f"{btts_prob:.0%}")
        with col_o:
            st.metric("Over 2.5", f"{ou['over']:.0%}")
        with col_lam:
            st.metric(
                f"Lambda {home_name} / {away_name}",
                f"{lambda_home_adj:.2f} / {lambda_away_adj:.2f}",
            )

        # --- Central: Score matrix + top scores ---
        st.subheader("Matriz de Placares")
        fig_mat = render_score_matrix(score_mat, home_name, away_name)
        st.plotly_chart(fig_mat, use_container_width=True)

        st.subheader("Placares Mais Provaveis")
        if top_scores:
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

        # --- Context: Narrative panel ---
        st.subheader("Analise do Agente")
        render_narrative_panel(
            narrative=narrative_text,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            lambda_home_adj=lambda_home_adj,
            lambda_away_adj=lambda_away_adj,
            adjustment_log=adj_log,
        )

        # --- Technical section (expandable) ---
        with st.expander("Detalhes Tecnicos"):
            st.markdown(f"- Lambda original: Casa={lambda_home:.3f}, Fora={lambda_away:.3f}")
            st.markdown(f"- Lambda ajustado: Casa={lambda_home_adj:.3f}, Fora={lambda_away_adj:.3f}")
            if not match_pred.empty:
                st.markdown(f"- Gerado em: {match_pred.iloc[0].get('generated_at', '?')}")

    except Exception as _exc:
        st.error(f"Erro ao renderizar a analise: {_exc}")
        logger.exception("Erro na pagina Analise de Jogo")

# ===========================================================================
# PAGE 3 — Performance do Modelo
# ===========================================================================

elif page == "Performance do Modelo":
    st.title("Performance do Modelo")

    preds_results = load_predictions_with_results()

    if preds_results.empty:
        st.info("Sem previsoes com resultados disponiveis para avaliacao.")
    else:
        from evaluation.metrics import compute_all_metrics
        from model.calibration import plot_reliability_diagram

        # Overall metrics
        metrics_df = compute_all_metrics(preds_results)
        render_metrics_panel(metrics_df)

        # Last 10 rounds metrics
        if "round" in preds_results.columns:
            rounds = preds_results["round"].dropna().unique()
            rounds_sorted = sorted(rounds, key=_round_sort_key, reverse=True)
            last_10 = rounds_sorted[:10]
            recent = preds_results[preds_results["round"].isin(last_10)]
            if not recent.empty:
                st.subheader("Ultimas 10 Rodadas")
                recent_metrics = compute_all_metrics(recent)
                render_metrics_panel(recent_metrics)

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
                    title="Reliability Diagram — Vitoria Mandante",
                )
                st.plotly_chart(fig_rel, use_container_width=True)

        # Walk-forward RPS
        st.subheader("RPS Walk-Forward")
        finished = load_finished_matches()
        if not finished.empty and len(finished) > 200:
            with st.spinner("Calculando backtest ..."):
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
        st.subheader("Ultimas 20 Previsoes vs Resultado")
        display_cols = [
            c for c in [
                "match_id", "match_date", "prob_home", "prob_draw", "prob_away",
                "home_goals", "away_goals",
            ]
            if c in preds_results.columns
        ]
        st.dataframe(preds_results[display_cols].head(20), use_container_width=True)

# ===========================================================================
# PAGE 4 — Parametros
# ===========================================================================

elif page == "Parametros":
    st.title("Parametros do Modelo")

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
        st.info("Sem parametros disponiveis. Treine o modelo primeiro.")
    else:
        if "team_id" in params_df.columns:
            params_df["team"] = params_df["team_id"].map(teams)

        # Rankings with error bars
        col_atk, col_def = st.columns(2)
        with col_atk:
            st.subheader("Ranking — Forca de Ataque")
            atk_df = params_df.sort_values("attack", ascending=False)
            if "attack_std" in atk_df.columns and "team" in atk_df.columns:
                fig_atk = go.Figure(go.Bar(
                    x=atk_df["attack"].tolist(),
                    y=atk_df["team"].tolist(),
                    orientation="h",
                    error_x=dict(type="data", array=atk_df["attack_std"].fillna(0).tolist()),
                    marker_color="royalblue",
                ))
                fig_atk.update_layout(
                    yaxis={"autorange": "reversed"}, template="plotly_white", height=500
                )
                st.plotly_chart(fig_atk, use_container_width=True)
            else:
                atk_cols = [c for c in ["team", "attack"] if c in params_df.columns]
                st.dataframe(atk_df[atk_cols], use_container_width=True)

        with col_def:
            st.subheader("Ranking — Forca de Defesa (menor = melhor)")
            def_df = params_df.sort_values("defense", ascending=True)
            if "defense_std" in def_df.columns and "team" in def_df.columns:
                fig_def = go.Figure(go.Bar(
                    x=def_df["defense"].tolist(),
                    y=def_df["team"].tolist(),
                    orientation="h",
                    error_x=dict(type="data", array=def_df["defense_std"].fillna(0).tolist()),
                    marker_color="tomato",
                ))
                fig_def.update_layout(
                    yaxis={"autorange": "reversed"}, template="plotly_white", height=500
                )
                st.plotly_chart(fig_def, use_container_width=True)
            else:
                def_cols = [c for c in ["team", "defense"] if c in params_df.columns]
                st.dataframe(def_df[def_cols], use_container_width=True)

        # Violin plots for top 8 teams
        if "attack" in params_df.columns and "team" in params_df.columns:
            st.subheader("Distribuicoes Posteriores — Top 8 Times (Ataque)")
            top8 = params_df.nlargest(8, "attack")["team"].tolist()
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
                    for team_name in top8:
                        mean_row = params_df[params_df["team"] == team_name]
                        std_row = std_df[std_df["team"] == team_name]
                        if mean_row.empty or std_row.empty:
                            continue
                        mu = float(mean_row["attack"].iloc[0])
                        sigma = float(std_row["attack"].iloc[0])
                        samples = np.random.normal(mu, sigma, 500)
                        fig_violin.add_trace(
                            go.Violin(
                                y=samples.tolist(), name=team_name,
                                box_visible=True, meanline_visible=True,
                            )
                        )
                    fig_violin.update_layout(
                        title="Posterior Ataque — Top 8 Times",
                        template="plotly_white",
                        yaxis_title="Parametro de Ataque",
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
            except Exception as exc:
                logger.warning("Could not render violin plots: %s", exc)

        # Dynamic parameter evolution
        st.subheader("Evolucao Temporal dos Parametros")
        trace_dir = Path(__file__).parent.parent / "traces"
        trace_files = sorted(trace_dir.glob("*.nc"), reverse=True) if trace_dir.exists() else []
        if trace_files:
            try:
                import arviz as az
                from model.dynamic import plot_param_evolution

                selected_team_param = st.selectbox(
                    "Time para evolucao",
                    list(teams.values())[:20] if teams else [],
                )
                if selected_team_param:
                    team_idx = next(
                        (k for k, v in teams.items() if v == selected_team_param), None
                    )
                    if team_idx is not None:
                        st.caption(f"(Funcionalidade requer modelo dinamico treinado)")
            except ImportError:
                st.info("arviz nao disponivel para trace plots.")
        else:
            st.info("Nenhum trace MCMC encontrado em `traces/`.")

        # MCMC trace plots
        st.subheader("Trace Plots MCMC")
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
                st.warning(f"Nao foi possivel carregar o trace: {exc}")

# ===========================================================================
# PAGE 5 — Agentes
# ===========================================================================

elif page == "Agentes":
    st.title("Agentes de Inteligencia")

    # --- Calibration agent history ---
    st.subheader("Historico do Agente de Calibracao")
    cal_history = load_calibration_history()

    if cal_history.empty:
        st.info("Sem historico de calibracao. Execute a analise apos uma rodada finalizada.")
    else:
        for _, cal_row in cal_history.head(10).iterrows():
            with st.expander(
                f"Rodada {cal_row.get('round_analyzed', '?')} — "
                f"Temporada {cal_row.get('season', '?')} — "
                f"{'Aplicado' if cal_row.get('applied') else 'Pendente'}"
            ):
                reasoning = cal_row.get("agent_reasoning", "")
                if reasoning:
                    st.markdown(reasoning)

                adjustments = cal_row.get("suggested_adjustments")
                if adjustments:
                    if isinstance(adjustments, str):
                        adjustments = json.loads(adjustments)
                    if isinstance(adjustments, list):
                        for adj in adjustments:
                            st.markdown(
                                f"- **{adj.get('parametro', '?')}**: {adj.get('ajuste', '')} "
                                f"(magnitude={adj.get('magnitude', '?')}, "
                                f"confianca={adj.get('confianca', '?')})"
                            )

                error_patterns = cal_row.get("error_patterns")
                if error_patterns:
                    if isinstance(error_patterns, str):
                        error_patterns = json.loads(error_patterns)
                    st.json(error_patterns)

    # --- Context agent impact ---
    st.subheader("Impacto do Agente de Contexto")
    predictions_df = load_predictions()
    if not predictions_df.empty and "lambda_home" in predictions_df.columns:
        has_adjusted = "lambda_home_adjusted" in predictions_df.columns
        if has_adjusted:
            preds_with_adj = predictions_df.dropna(subset=["lambda_home_adjusted"]).copy()
            if not preds_with_adj.empty:
                preds_with_adj["delta_home"] = (
                    preds_with_adj["lambda_home_adjusted"] - preds_with_adj["lambda_home"]
                )
                preds_with_adj["delta_away"] = (
                    preds_with_adj["lambda_away_adjusted"] - preds_with_adj["lambda_away"]
                )

                fig_impact = go.Figure()
                fig_impact.add_trace(go.Histogram(
                    x=preds_with_adj["delta_home"].tolist(),
                    name="Delta Lambda Casa",
                    opacity=0.7,
                ))
                fig_impact.add_trace(go.Histogram(
                    x=preds_with_adj["delta_away"].tolist(),
                    name="Delta Lambda Fora",
                    opacity=0.7,
                ))
                fig_impact.update_layout(
                    title="Distribuicao dos Ajustes Contextuais em Lambda",
                    xaxis_title="Delta Lambda",
                    yaxis_title="Frequencia",
                    barmode="overlay",
                    template="plotly_white",
                )
                st.plotly_chart(fig_impact, use_container_width=True)

                st.markdown(
                    f"Media do ajuste: Casa={preds_with_adj['delta_home'].mean():+.4f}, "
                    f"Fora={preds_with_adj['delta_away'].mean():+.4f}"
                )
            else:
                st.info("Sem previsoes com lambdas ajustados.")
        else:
            st.info("Previsoes nao contem lambdas ajustados.")
    else:
        st.info("Sem dados de previsao para analise de impacto.")

    # --- Context confidence log ---
    st.subheader("Confianca das Leituras Contextuais")
    try:
        from config import SUPABASE_URL, SUPABASE_KEY
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        ctx_resp = (
            client.table("match_context")
            .select("match_id, generated_at, agent_model, processed_context")
            .order("generated_at", desc=True)
            .limit(20)
            .execute()
        )
        if ctx_resp.data:
            for ctx_row in ctx_resp.data:
                pc = ctx_row.get("processed_context", {})
                if isinstance(pc, str):
                    pc = json.loads(pc)
                h_conf = pc.get("home", {}).get("confianca", 0)
                a_conf = pc.get("away", {}).get("confianca", 0)
                st.caption(
                    f"Match {ctx_row.get('match_id')} — "
                    f"Confianca: Casa={h_conf:.0%}, Fora={a_conf:.0%} — "
                    f"{ctx_row.get('generated_at', '')[:16]}"
                )
        else:
            st.info("Sem dados de contexto registrados.")
    except Exception as exc:
        st.info(f"Nao foi possivel carregar contexto: {exc}")

    # --- Run calibration button ---
    st.subheader("Executar Analise de Erros")
    if st.button("Rodar Analise de Erros da Ultima Rodada"):
        try:
            from agents.calibration_agent import (
                analyze_round_errors,
                generate_calibration_insights,
                save_and_return_insights,
            )
            preds = load_predictions_with_results()
            if preds.empty:
                st.warning("Sem previsoes com resultados para analisar.")
            else:
                if "round" in preds.columns:
                    latest_round = sorted(
                        preds["round"].dropna().unique(),
                        key=_round_sort_key,
                        reverse=True,
                    )[0]
                    round_preds = preds[preds["round"] == latest_round]
                    finished = load_finished_matches()

                    with st.spinner("Analisando erros..."):
                        errors = analyze_round_errors(round_preds, finished, latest_round)
                        insights = generate_calibration_insights(errors, round_num=latest_round)

                        repo = ModelRepository()
                        season_val = int(round_preds["season"].iloc[0]) if "season" in round_preds.columns else 2024
                        save_and_return_insights(errors, insights, latest_round, season_val, repo)

                    st.success(f"Analise completa para {_clean_round(latest_round)}.")
                    st.markdown(insights.get("resumo", ""))
                    if insights.get("padroes_identificados"):
                        st.markdown("**Padroes:**")
                        for p in insights["padroes_identificados"]:
                            st.markdown(f"- {p}")
        except Exception as exc:
            st.error(f"Erro na analise: {exc}")
