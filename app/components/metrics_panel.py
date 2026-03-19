"""Streamlit component for displaying model evaluation metrics."""

import pandas as pd
import streamlit as st


def render_metrics_panel(metrics_df: pd.DataFrame) -> None:
    """Render a row of metric cards from a metrics summary DataFrame.

    Args:
        metrics_df: Single-row DataFrame with columns: brier_score, rps,
            log_loss, n_samples (as returned by compute_all_metrics()).
    """
    if metrics_df.empty:
        st.warning("Sem métricas disponíveis.")
        return

    row = metrics_df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Brier Score",
            f"{row.get('brier_score', 'N/A'):.4f}" if row.get("brier_score") is not None else "N/A",
            help="Menor é melhor (0 = perfeito, 2 = pior).",
        )
    with col2:
        st.metric(
            "RPS",
            f"{row.get('rps', 'N/A'):.4f}" if row.get("rps") is not None else "N/A",
            help="Ranked Probability Score — menor é melhor.",
        )
    with col3:
        st.metric(
            "Log-Loss",
            f"{row.get('log_loss', 'N/A'):.4f}" if row.get("log_loss") is not None else "N/A",
            help="Log-loss multinomial — menor é melhor.",
        )
    with col4:
        st.metric(
            "Amostras",
            int(row.get("n_samples", 0)),
            help="Número de partidas avaliadas.",
        )
