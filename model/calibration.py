"""Post-model probability calibration.

Provides isotonic / Platt calibration of raw model probabilities and
a Plotly reliability diagram.
"""

import logging
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

_CalibrationMethod = Literal["isotonic", "platt"]


# ---------------------------------------------------------------------------
# Training / fitting
# ---------------------------------------------------------------------------


def calibrate_probabilities(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: _CalibrationMethod = "isotonic",
) -> IsotonicRegression | LogisticRegression:
    """Fit a calibration model on held-out data.

    Args:
        y_true: Binary outcome array (1 = event occurred, 0 = did not).
            For 1X2 calibration, call this separately for each class.
        y_pred_proba: Raw probability estimates from the forecast model,
            same length as y_true.
        method: 'isotonic' (Isotonic Regression) or 'platt'
            (Logistic Regression, a.k.a. Platt Scaling).

    Returns:
        Fitted calibrator object with a predict() method.

    Raises:
        ValueError: When an unknown method is requested.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.clip(np.asarray(y_pred_proba).ravel(), 1e-7, 1 - 1e-7)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_pred_proba, y_true)
    elif method == "platt":
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
    else:
        raise ValueError(
            f"Unknown calibration method '{method}'. Use 'isotonic' or 'platt'."
        )

    logger.info(
        "Fitted %s calibrator on %d samples.", method, len(y_true)
    )
    return calibrator


# ---------------------------------------------------------------------------
# Applying calibration
# ---------------------------------------------------------------------------


def apply_calibration(
    calibrator: IsotonicRegression | LogisticRegression,
    probs: np.ndarray,
) -> np.ndarray:
    """Apply a fitted calibrator to new probabilities.

    After calibration the three 1X2 probabilities are re-normalised so
    they still sum to 1.0.

    Args:
        calibrator: Fitted calibrator from calibrate_probabilities().
        probs: Array of shape (N,) or (N, 3). When shape is (N, 3), each
            column is calibrated independently and then normalised row-wise.

    Returns:
        Calibrated probabilities with the same shape as input.
    """
    probs = np.asarray(probs, dtype=float)

    def _calibrate_1d(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        if isinstance(calibrator, IsotonicRegression):
            return np.clip(calibrator.predict(p), 0.0, 1.0)
        else:  # LogisticRegression
            return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]

    if probs.ndim == 1:
        return _calibrate_1d(probs)

    if probs.ndim == 2 and probs.shape[1] == 3:
        calibrated = np.column_stack(
            [_calibrate_1d(probs[:, k]) for k in range(3)]
        )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return calibrated / row_sums

    raise ValueError(
        f"Expected probs of shape (N,) or (N, 3), got {probs.shape}."
    )


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> go.Figure:
    """Generate a Plotly calibration / reliability diagram.

    Args:
        y_true: Binary outcome array.
        y_pred_proba: Raw or calibrated probability estimates.
        n_bins: Number of equally-spaced probability bins.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.clip(np.asarray(y_pred_proba).ravel(), 1e-7, 1 - 1e-7)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
    )

    fig = go.Figure()

    # Perfect calibration reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Perfect calibration",
        )
    )

    # Model calibration curve
    fig.add_trace(
        go.Scatter(
            x=mean_predicted_value.tolist(),
            y=fraction_of_positives.tolist(),
            mode="lines+markers",
            marker={"size": 8},
            line={"color": "royalblue"},
            name="Model",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend={"x": 0.01, "y": 0.99},
        template="plotly_white",
    )

    return fig
