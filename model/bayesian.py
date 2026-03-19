"""Bayesian Dixon-Coles model via PyMC.

Implements a hierarchical Bayesian version of the Dixon-Coles model using
NUTS sampling, saving the posterior trace as a NetCDF file for later use.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from config import (
    DECAY_RATE,
    HOME_ADVANTAGE_PRIOR_MEAN,
    MCMC_CHAINS,
    MCMC_DRAWS,
    MCMC_TUNE,
)
from model.dixon_coles import dixon_coles_correction

logger = logging.getLogger(__name__)

_TRACE_DIR = Path(__file__).parent.parent / "traces"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_bayesian_model(
    matches_df: pd.DataFrame,
    team_index: dict[int, int],
    decay_rate: float = DECAY_RATE,
) -> tuple[pm.Model, dict[str, Any]]:
    """Construct the PyMC Dixon-Coles model.

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date.
        team_index: Mapping from team_id (int) to contiguous integer index.
        decay_rate: Exponential decay factor for temporal weighting.

    Returns:
        Tuple of (pymc.Model, data_dict) where data_dict contains the
        processed arrays and weights used inside the model.
    """
    df = matches_df.dropna(
        subset=["home_goals", "away_goals", "home_team_id", "away_team_id", "match_date"]
    ).copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    df = df.sort_values("match_date").reset_index(drop=True)

    now = datetime.now(tz=timezone.utc)
    df["days_ago"] = df["match_date"].apply(lambda d: max((now - d).days, 0))
    df["weight"] = decay_rate ** (df["days_ago"] / 30.0)

    home_idx = df["home_team_id"].map(team_index).values.astype(int)
    away_idx = df["away_team_id"].map(team_index).values.astype(int)
    home_goals = df["home_goals"].values.astype(int)
    away_goals = df["away_goals"].values.astype(int)
    weights = df["weight"].values.astype(float)

    n_teams = len(team_index)
    n_matches = len(df)

    # Precompute Dixon-Coles correction masks — will be applied via pm.Potential
    # We model rho as a scalar latent variable and compute tau analytically.

    data_dict = {
        "home_idx": home_idx,
        "away_idx": away_idx,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "weights": weights,
        "n_teams": n_teams,
        "n_matches": n_matches,
    }

    with pm.Model() as model:
        # ── Priors ──────────────────────────────────────────────────────────
        home_advantage = pm.Normal(
            "home_advantage", mu=HOME_ADVANTAGE_PRIOR_MEAN, sigma=0.2
        )
        rho = pm.Uniform("rho", lower=-0.99, upper=0.0)
        intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)

        attack = pm.Normal("attack", mu=0.0, sigma=1.0, shape=n_teams)
        defense = pm.Normal("defense", mu=0.0, sigma=1.0, shape=n_teams)

        # ── Expected goals ───────────────────────────────────────────────────
        mu_home = pm.math.exp(
            intercept
            + home_advantage
            + attack[home_idx]
            + defense[away_idx]
        )
        mu_away = pm.math.exp(
            intercept + attack[away_idx] + defense[home_idx]
        )

        # ── Poisson log-likelihood ───────────────────────────────────────────
        home_logp = pm.logp(pm.Poisson.dist(mu=mu_home), home_goals)
        away_logp = pm.logp(pm.Poisson.dist(mu=mu_away), away_goals)

        # ── Dixon-Coles correction via pm.Potential ──────────────────────────
        # Compute tau correction numerically for each observation
        tau_vals = _compute_tau_tensor(
            home_goals, away_goals, mu_home, mu_away, rho
        )

        weighted_ll = weights * (home_logp + away_logp + pm.math.log(tau_vals))
        pm.Potential("weighted_likelihood", weighted_ll.sum())

    logger.info(
        "Built Bayesian model: %d teams, %d matches.", n_teams, n_matches
    )
    return model, data_dict


def _compute_tau_tensor(
    x: np.ndarray,
    y: np.ndarray,
    lambda_h: Any,
    lambda_a: Any,
    rho: Any,
) -> Any:
    """Compute Dixon-Coles tau correction as a PyTensor expression.

    Args:
        x: Array of home goals (integer).
        y: Array of away goals (integer).
        lambda_h: PyTensor expression for home expected goals.
        lambda_a: PyTensor expression for away expected goals.
        rho: PyTensor scalar for the dependency parameter.

    Returns:
        PyTensor expression of shape (n_matches,) with tau values.
    """
    import pytensor.tensor as pt

    # We construct tau as a piecewise expression using conditions
    is_00 = pt.and_(pt.eq(x, 0), pt.eq(y, 0))
    is_10 = pt.and_(pt.eq(x, 1), pt.eq(y, 0))
    is_01 = pt.and_(pt.eq(x, 0), pt.eq(y, 1))
    is_11 = pt.and_(pt.eq(x, 1), pt.eq(y, 1))

    tau = pt.switch(
        is_00, 1.0 - lambda_h * lambda_a * rho,
        pt.switch(
            is_10, 1.0 + lambda_a * rho,
            pt.switch(
                is_01, 1.0 + lambda_h * rho,
                pt.switch(is_11, 1.0 - rho, 1.0),
            ),
        ),
    )
    # Clip to avoid log(0)
    return pt.clip(tau, 1e-6, np.inf)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_posterior(
    model: pm.Model,
    draws: int = MCMC_DRAWS,
    tune: int = MCMC_TUNE,
    chains: int = MCMC_CHAINS,
    trace_path: Path | None = None,
) -> az.InferenceData:
    """Run NUTS sampling on the provided model.

    Args:
        model: A PyMC model as returned by build_bayesian_model().
        draws: Number of posterior samples per chain.
        tune: Number of tuning steps per chain.
        chains: Number of independent Markov chains.
        trace_path: Optional path to save the NetCDF trace. Defaults to
            traces/trace_<timestamp>.nc inside the project root.

    Returns:
        ArviZ InferenceData object containing the posterior.
    """
    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    if trace_path is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        trace_path = _TRACE_DIR / f"trace_{ts}.nc"

    logger.info(
        "Starting NUTS sampling: draws=%d, tune=%d, chains=%d …",
        draws,
        tune,
        chains,
    )
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True,
        )

    idata.to_netcdf(str(trace_path))
    logger.info("Trace saved to %s", trace_path)
    return idata


# ---------------------------------------------------------------------------
# Posterior extraction
# ---------------------------------------------------------------------------


def get_posterior_means(
    idata: az.InferenceData,
    team_names: dict[int, str],
) -> pd.DataFrame:
    """Extract posterior mean and standard deviation for attack/defence.

    Args:
        idata: ArviZ InferenceData returned by sample_posterior().
        team_names: Mapping from contiguous integer index → team name string.

    Returns:
        DataFrame with columns: team, attack_mean, attack_std,
        defense_mean, defense_std, home_advantage.
    """
    posterior = idata.posterior  # type: ignore[attr-defined]

    attack_mean = posterior["attack"].mean(dim=["chain", "draw"]).values
    attack_std = posterior["attack"].std(dim=["chain", "draw"]).values
    defense_mean = posterior["defense"].mean(dim=["chain", "draw"]).values
    defense_std = posterior["defense"].std(dim=["chain", "draw"]).values
    home_adv = float(
        posterior["home_advantage"].mean(dim=["chain", "draw"]).values
    )

    records = []
    for idx, name in team_names.items():
        records.append(
            {
                "team": name,
                "attack_mean": float(attack_mean[idx]),
                "attack_std": float(attack_std[idx]),
                "defense_mean": float(defense_mean[idx]),
                "defense_std": float(defense_std[idx]),
                "home_advantage": home_adv,
            }
        )

    df = pd.DataFrame(records).sort_values("attack_mean", ascending=False)
    logger.info("Extracted posterior means for %d teams.", len(df))
    return df
