"""Plotly heatmap component for the score probability matrix."""

import numpy as np
import plotly.graph_objects as go


def render_score_matrix(
    score_matrix: list[list[float]] | np.ndarray,
    home_team: str,
    away_team: str,
    max_display: int = 7,
) -> go.Figure:
    """Build a Plotly heatmap of score probabilities.

    Args:
        score_matrix: 2-D array of shape (MAX_GOALS+1, MAX_GOALS+1) where
            element [i, j] = P(home=i, away=j).
        home_team: Display name for the home team (y-axis label).
        away_team: Display name for the away team (x-axis label).
        max_display: Maximum goals to show on each axis (truncated to
            min(max_display, score_matrix.shape[0] - 1)).

    Returns:
        Plotly Figure.
    """
    mat = np.asarray(score_matrix, dtype=float)
    display_size = min(max_display + 1, mat.shape[0])
    mat = mat[:display_size, :display_size]

    labels = [str(g) for g in range(display_size)]

    # Text annotations: percentage in each cell
    text = [
        [f"{mat[i, j] * 100:.1f}%" for j in range(display_size)]
        for i in range(display_size)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=mat.tolist(),
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            colorbar={"title": "Probabilidade"},
        )
    )

    # Find most probable score
    max_idx = np.unravel_index(mat.argmax(), mat.shape)

    # Highlight cells with prob > 10% and mark most probable
    for i in range(display_size):
        for j in range(display_size):
            if mat[i, j] > 0.10:
                is_max = (i, j) == max_idx
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5,
                    x1=j + 0.5,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    line={
                        "color": "gold" if is_max else "white",
                        "width": 3 if is_max else 2,
                    },
                )

    fig.update_layout(
        title=f"Distribuição de Placares — {home_team} x {away_team}",
        xaxis_title=f"Gols {away_team}",
        yaxis_title=f"Gols {home_team}",
        template="plotly_white",
        height=500,
    )

    return fig
