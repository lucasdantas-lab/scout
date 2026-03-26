"""Reusable Streamlit component for displaying a single match prediction."""

import streamlit as st

# SVG icons — Heroicons outline set (MIT)
_SVG_UP = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="none" '
    'viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M4.5 10.5 12 3m0 0 7.5 7.5M12 3v18"/>'
    "</svg>"
)
_SVG_DOWN = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="none" '
    'viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M19.5 13.5 12 21m0 0-7.5-7.5M12 21V3"/>'
    "</svg>"
)
_SVG_DRAW = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="none" '
    'viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M5 12h14"/>'
    "</svg>"
)


def render_match_card(
    match_date: str,
    home_team: str,
    away_team: str,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    prob_btts: float,
    prob_over25: float,
    actual_result: str | None = None,
    most_likely_score: str | None = None,
    context_absences: list[str] | None = None,
    context_confidence: float | None = None,
) -> None:
    """Render a match prediction card inside a bordered Streamlit container.

    Args:
        match_date: Human-readable match date/time string.
        home_team: Display name of the home team.
        away_team: Display name of the away team.
        prob_home: Probability of home win in [0, 1].
        prob_draw: Probability of draw in [0, 1].
        prob_away: Probability of away win in [0, 1].
        prob_btts: Probability of both teams scoring in [0, 1].
        prob_over25: Probability of over 2.5 goals in [0, 1].
        actual_result: Optional string like "2 x 1" shown when match is played.
        most_likely_score: Optional string like "1-0" for the top scoreline.
        context_absences: Optional list of key absences from context agent.
        context_confidence: Optional confidence of context info (0-1).
    """
    max_prob = max(prob_home, prob_draw, prob_away)
    if max_prob == prob_home:
        badge_bg = "#166534"   # dark green
        badge_icon = _SVG_UP
        badge_text = home_team
    elif max_prob == prob_draw:
        badge_bg = "#374151"   # dark gray
        badge_icon = _SVG_DRAW
        badge_text = "Empate"
    else:
        badge_bg = "#991b1b"   # dark red
        badge_icon = _SVG_DOWN
        badge_text = away_team

    badge_html = (
        f"<span style='display:inline-flex;align-items:center;gap:4px;"
        f"background:{badge_bg};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.75em;font-weight:600;"
        f"letter-spacing:0.03em'>"
        f"{badge_icon}{badge_text}</span>"
    )

    with st.container(border=True):
        st.caption(match_date)
        col_badge, col_title = st.columns([1, 4])
        with col_badge:
            st.markdown(badge_html, unsafe_allow_html=True)
        with col_title:
            title = f"**{home_team}** vs **{away_team}**"
            if actual_result:
                title += f"  —  `{actual_result}`"
            st.markdown(title)

        # Context line: absences detected
        if context_absences:
            absence_str = ", ".join(context_absences[:4])
            confidence_icon = "" if context_confidence and context_confidence >= 0.4 else " :warning:"
            st.caption(f"Desfalques: {absence_str}{confidence_icon}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Casa", f"{prob_home:.0%}")
            st.progress(prob_home)
        with col2:
            st.metric("Empate", f"{prob_draw:.0%}")
            st.progress(prob_draw)
        with col3:
            st.metric("Fora", f"{prob_away:.0%}")
            st.progress(prob_away)

        sub_parts = [
            f"BTTS: **{prob_btts:.0%}**",
            f"Over 2.5: **{prob_over25:.0%}**",
        ]
        if most_likely_score:
            sub_parts.append(f"Placar: **{most_likely_score}**")
        st.caption("  ·  ".join(sub_parts))
