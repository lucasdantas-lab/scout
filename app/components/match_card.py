"""Reusable Streamlit component for displaying a single match prediction."""

import streamlit as st


def render_match_card(
    match_date: str,
    home_team: str,
    away_team: str,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    prob_btts: float,
    prob_over25: float,
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
    """
    # Determine outcome badge
    max_prob = max(prob_home, prob_draw, prob_away)
    if max_prob == prob_home:
        badge_color = "green"
        badge_label = f"▲ {home_team}"
    elif max_prob == prob_draw:
        badge_color = "gray"
        badge_label = "⇔ Empate"
    else:
        badge_color = "red"
        badge_label = f"▼ {away_team}"

    with st.container(border=True):
        st.caption(match_date)
        col_badge, col_title = st.columns([1, 4])
        with col_badge:
            st.markdown(
                f"<span style='background:{badge_color};color:white;"
                f"padding:2px 8px;border-radius:4px;font-size:0.8em'>"
                f"{badge_label}</span>",
                unsafe_allow_html=True,
            )
        with col_title:
            st.markdown(f"**{home_team}** vs **{away_team}**")

        # Probability bars
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

        st.caption(
            f"BTTS: **{prob_btts:.0%}**  ·  Over 2.5: **{prob_over25:.0%}**"
        )
