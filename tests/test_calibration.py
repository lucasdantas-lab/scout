"""Unit tests for model/calibration.py."""

import numpy as np
import pytest

from model.calibration import (
    apply_calibration,
    calibrate_probabilities,
    plot_reliability_diagram,
)


class TestCalibrateProabilities:
    def test_isotonic_fits(self) -> None:
        """Isotonic calibrator should fit without errors."""
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.uniform(0.1, 0.9, size=100)

        cal = calibrate_probabilities(y_true, y_pred, method="isotonic")
        assert hasattr(cal, "predict")

    def test_platt_fits(self) -> None:
        """Platt (logistic) calibrator should fit without errors."""
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.uniform(0.1, 0.9, size=100)

        cal = calibrate_probabilities(y_true, y_pred, method="platt")
        assert hasattr(cal, "predict_proba")

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrate_probabilities(np.array([1]), np.array([0.5]), method="unknown")


class TestApplyCalibration:
    def test_1d_output(self) -> None:
        """Apply calibration to 1D input."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.uniform(0.1, 0.9, size=100)

        cal = calibrate_probabilities(y_true, y_pred, method="isotonic")
        result = apply_calibration(cal, np.array([0.3, 0.5, 0.7]))
        assert result.shape == (3,)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_3col_normalised(self) -> None:
        """Apply calibration to (N, 3) input; rows should sum to 1."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.uniform(0.1, 0.9, size=200)

        cal = calibrate_probabilities(y_true, y_pred, method="isotonic")
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
        result = apply_calibration(cal, probs)
        assert result.shape == (2, 3)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestPlotReliabilityDiagram:
    def test_returns_figure(self) -> None:
        """Should return a plotly Figure."""
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.uniform(0.1, 0.9, size=50)

        fig = plot_reliability_diagram(y_true, y_pred, n_bins=5)
        assert fig is not None
        assert hasattr(fig, "data")
