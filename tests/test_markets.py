"""Unit tests for model/markets.py."""

import numpy as np
import pytest

from model.markets import (
    compute_1x2,
    compute_btts,
    compute_exact_scores,
    compute_over_under,
    compute_score_matrix,
)


# ---------------------------------------------------------------------------
# compute_score_matrix
# ---------------------------------------------------------------------------


class TestComputeScoreMatrix:
    def test_shape(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1, max_goals=8)
        assert mat.shape == (9, 9)

    def test_sums_to_one(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        assert abs(mat.sum() - 1.0) < 1e-6

    def test_non_negative(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        assert (mat >= 0).all()

    def test_higher_lambda_shifts_mass(self) -> None:
        """Higher λ_home → more mass on rows where home > away."""
        mat_high = compute_score_matrix(3.0, 0.5, 0.0)
        mat_low = compute_score_matrix(0.5, 3.0, 0.0)
        n = mat_high.shape[0]
        i_idx, j_idx = np.tril_indices(n, k=-1)
        assert mat_high[i_idx, j_idx].sum() > mat_low[i_idx, j_idx].sum()

    def test_custom_max_goals(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1, max_goals=5)
        assert mat.shape == (6, 6)


# ---------------------------------------------------------------------------
# compute_1x2
# ---------------------------------------------------------------------------


class TestCompute1x2:
    def test_sums_to_one(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        probs = compute_1x2(mat)
        total = probs["home"] + probs["draw"] + probs["away"]
        assert abs(total - 1.0) < 1e-6

    def test_keys_present(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        probs = compute_1x2(mat)
        assert {"home", "draw", "away"} == set(probs.keys())

    def test_all_probabilities_in_range(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        probs = compute_1x2(mat)
        for v in probs.values():
            assert 0.0 <= v <= 1.0

    def test_home_favoured_when_higher_lambda(self) -> None:
        mat = compute_score_matrix(2.5, 0.8, 0.0)
        probs = compute_1x2(mat)
        assert probs["home"] > probs["away"]


# ---------------------------------------------------------------------------
# compute_btts
# ---------------------------------------------------------------------------


class TestComputeBTTS:
    def test_range(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        btts = compute_btts(mat)
        assert 0.0 <= btts <= 1.0

    def test_increases_with_both_lambdas(self) -> None:
        mat_low = compute_score_matrix(0.5, 0.5, 0.0)
        mat_high = compute_score_matrix(2.5, 2.5, 0.0)
        assert compute_btts(mat_high) > compute_btts(mat_low)


# ---------------------------------------------------------------------------
# compute_over_under
# ---------------------------------------------------------------------------


class TestComputeOverUnder:
    def test_sums_to_one(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        ou = compute_over_under(mat, line=2.5)
        assert abs(ou["over"] + ou["under"] - 1.0) < 1e-6

    def test_over_increases_with_lambda(self) -> None:
        mat_low = compute_score_matrix(0.7, 0.7, 0.0)
        mat_high = compute_score_matrix(2.5, 2.5, 0.0)
        assert compute_over_under(mat_high)["over"] > compute_over_under(mat_low)["over"]


# ---------------------------------------------------------------------------
# compute_exact_scores
# ---------------------------------------------------------------------------


class TestComputeExactScores:
    def test_returns_top_n(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        scores = compute_exact_scores(mat, top_n=12)
        assert len(scores) == 12

    def test_sorted_descending(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        scores = compute_exact_scores(mat, top_n=12)
        probs = [s["prob"] for s in scores]
        assert probs == sorted(probs, reverse=True)

    def test_score_format(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        scores = compute_exact_scores(mat, top_n=5)
        for s in scores:
            parts = s["score"].split("-")
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)

    def test_probabilities_sum_leq_one(self) -> None:
        mat = compute_score_matrix(1.5, 1.1, -0.1)
        scores = compute_exact_scores(mat, top_n=12)
        assert sum(s["prob"] for s in scores) <= 1.0 + 1e-6
