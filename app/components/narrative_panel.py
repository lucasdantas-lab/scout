"""Streamlit component for displaying Claude-generated match narratives."""

from typing import Any

import streamlit as st


def render_narrative_panel(
    narrative: str | None,
    context_dict: dict[str, Any] | None = None,
    lambda_home: float | None = None,
    lambda_away: float | None = None,
    lambda_home_adj: float | None = None,
    lambda_away_adj: float | None = None,
    adjustment_log: list[str] | None = None,
    generated_at: str | None = None,
    on_regenerate: Any | None = None,
) -> None:
    """Render the narrative analysis panel with expandable technical details.

    Args:
        narrative: The Claude-generated narrative text.
        context_dict: Processed context from context_agent with 'home'
            and 'away' keys.
        lambda_home: Original home expected goals.
        lambda_away: Original away expected goals.
        lambda_home_adj: Adjusted home expected goals.
        lambda_away_adj: Adjusted away expected goals.
        adjustment_log: List of adjustment descriptions.
        generated_at: Timestamp string of when the narrative was generated.
        on_regenerate: Optional callback for the regenerate button.
    """
    if not narrative:
        st.info("Narrativa ainda nao gerada para este jogo.")
        return

    # Determine if context was meaningful
    has_context = False
    if context_dict:
        for side in ("home", "away"):
            side_ctx = context_dict.get(side, {})
            if (
                side_ctx.get("ausencias_confirmadas")
                or side_ctx.get("lambda_delta", 0) != 0
            ):
                has_context = True
                break

    # Context indicator badge
    if has_context:
        st.markdown(
            "<span style='background:#166534;color:white;padding:2px 8px;"
            "border-radius:4px;font-size:0.75em'>Com contexto de desfalques</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='background:#6b7280;color:white;padding:2px 8px;"
            "border-radius:4px;font-size:0.75em'>Sem info de desfalques</span>",
            unsafe_allow_html=True,
        )

    # Main narrative
    st.markdown(narrative)

    # Timestamp
    if generated_at:
        st.caption(f"Gerado em: {generated_at}")

    # Expandable technical details
    with st.expander("Ver detalhes tecnicos"):
        if lambda_home is not None and lambda_away is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Lambda Casa (original)",
                    f"{lambda_home:.3f}",
                )
                if lambda_home_adj is not None:
                    delta = lambda_home_adj - lambda_home
                    st.metric(
                        "Lambda Casa (ajustado)",
                        f"{lambda_home_adj:.3f}",
                        delta=f"{delta:+.3f}",
                    )
            with col2:
                st.metric(
                    "Lambda Fora (original)",
                    f"{lambda_away:.3f}",
                )
                if lambda_away_adj is not None:
                    delta = lambda_away_adj - lambda_away
                    st.metric(
                        "Lambda Fora (ajustado)",
                        f"{lambda_away_adj:.3f}",
                        delta=f"{delta:+.3f}",
                    )

        # Adjustment log
        if adjustment_log:
            st.markdown("**Ajustes aplicados:**")
            for adj in adjustment_log:
                st.markdown(f"- `{adj}`")

        # Context details
        if context_dict:
            for side_label, side_key in [("Mandante", "home"), ("Visitante", "away")]:
                side_ctx = context_dict.get(side_key, {})
                if side_ctx:
                    st.markdown(f"**{side_label}:**")
                    absences = side_ctx.get("ausencias_confirmadas", [])
                    doubts = side_ctx.get("duvidas", [])
                    confirmed = side_ctx.get("confirmados_importantes", [])
                    conf = side_ctx.get("confianca", 0)
                    notes = side_ctx.get("notas", "")

                    if absences:
                        st.markdown(f"- Ausencias: {', '.join(absences)}")
                    if doubts:
                        st.markdown(f"- Duvidas: {', '.join(doubts)}")
                    if confirmed:
                        st.markdown(f"- Confirmados: {', '.join(confirmed)}")
                    st.markdown(
                        f"- Delta: {side_ctx.get('lambda_delta', 0):+.2f} "
                        f"(confianca: {conf:.0%})"
                    )
                    if notes:
                        st.caption(notes)

    # Regenerate button
    if on_regenerate:
        if st.button("Regenerar narrativa", key=f"regen_{id(narrative)}"):
            on_regenerate()
