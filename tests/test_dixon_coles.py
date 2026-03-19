"""Unit tests for model/dixon_coles.py."""

import numpy as np
import pytest

from model.dixon_coles import (
    dixon_coles_correction,
    dixon_coles_log_likelihood,
    fit_dixon_coles_mle,
)


# ---------------------------------------------------------------------------
# dixon_coles_correction
# ---------------------------------------------------------------------------


class TestDixonColesCorrection:
    def test_zero_zero(self) -> None:
        tau = dixon_coles_correction(0, 0, lambda_home=1.5, lambda_away=1.0, rho=-0.1)
        expected = 1.0 - 1.5 * 1.0 * (-0.1)
        assert abs(tau - expected) < 1e-9

    def test_one_zero(self) -> None:
        tau = dixon_coles_correction(1, 0, lambda_home=1.5, lambda_away=1.0, rho=-0.1)
        expected = 1.0 + 1.0 * (-0.1)
        assert abs(tau - expected) < 1e-9

    def test_zero_one(self) -> None:
        tau = dixon_coles_correction(0, 1, lambda_home=1.5, lambda_away=1.0, rho=-0.1)
        expected = 1.0 + 1.5 * (-0.1)
        assert abs(tau - expected) < 1e-9

    def test_one_one(self) -> None:
        tau = dixon_coles_correction(1, 1, lambda_home=1.5, lambda_away=1.0, rho=-0.1)
        expected = 1.0 - (-0.1)
        assert abs(tau - expected) < 1e-9

    def test_high_score_returns_one(self) -> None:
        for x, y in [(2, 0), (0, 2), (3, 3), (5, 1)]:
            assert dixon_coles_correction(x, y, 1.5, 1.0, -0.1) == 1.0

    def test_rho_zero_returns_one(self) -> None:
        """When rho=0, correction is always 1 for all low scores."""
        for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            assert dixon_coles_correction(x, y, 1.5, 1.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# dixon_coles_log_likelihood
# ---------------------------------------------------------------------------


class TestLogLikelihood:
    def test_returns_scalar(self, two_team_matches) -> None:
        team_list = [1, 2]
        n = len(team_list)
        params = np.zeros(2 * n + 2)
        params[2 * n] = 0.3   # home_advantage
        params[2 * n + 1] = -0.1  # rho

        result = dixon_coles_log_likelihood(params, two_team_matches, team_list)
        assert isinstance(result, float)

    def test_positive_result(self, two_team_matches) -> None:
        """Negative log-likelihood should be positive (we minimise it)."""
        team_list = [1, 2]
        n = len(team_list)
        params = np.zeros(2 * n + 2)
        params[2 * n] = 0.3
        params[2 * n + 1] = -0.1

        result = dixon_coles_log_likelihood(params, two_team_matches, team_list)
        assert result > 0


# ---------------------------------------------------------------------------
# fit_dixon_coles_mle
# ---------------------------------------------------------------------------


class TestFitDixonColesMLE:
    def test_returns_expected_keys(self, sample_matches) -> None:
        result = fit_dixon_coles_mle(sample_matches)
        assert "team_params" in result
        assert "home_advantage" in result
        assert "rho" in result
        assert "success" in result

    def test_rho_in_bounds(self, sample_matches) -> None:
        result = fit_dixon_coles_mle(sample_matches)
        assert -0.99 <= result["rho"] <= 0.0

    def test_all_teams_present(self, sample_matches) -> None:
        result = fit_dixon_coles_mle(sample_matches)
        team_ids = set(sample_matches["home_team_id"]).union(
            sample_matches["away_team_id"]
        )
        assert set(result["team_params"].keys()) == team_ids

    def test_attack_mean_near_zero(self, sample_matches) -> None:
        """Identification constraint: mean of attack params ≈ 0."""
        result = fit_dixon_coles_mle(sample_matches)
        attacks = [v["attack"] for v in result["team_params"].values()]
        assert abs(np.mean(attacks)) < 0.1

    def test_raises_with_single_team(self) -> None:
        import pandas as pd

        bad_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "season": 2023,
                    "round": "1",
                    "match_date": pd.Timestamp("2023-01-01", tz="UTC"),
                    "home_team_id": 1,
                    "away_team_id": 1,
                    "home_goals": 1,
                    "away_goals": 0,
                    "status": "FT",
                    "venue": "X",
                }
            ]
        )
        with pytest.raises(ValueError, match="at least 2 teams"):
            fit_dixon_coles_mle(bad_df)
