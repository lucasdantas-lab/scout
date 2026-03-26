"""Bayesian Dixon-Coles model via PyMC.

Implements a hierarchical Bayesian version of the Dixon-Coles model using
NUTS sampling, with squad strength and form covariates, saving the posterior
trace as a NetCDF file for later use.
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
    use_xg: bool = False,
) -> tuple[pm.Model, dict[str, Any]]:
    """Construct the PyMC Dixon-Coles model with covariates.

    Includes squad strength and form as covariates in the expected goals
    formula. When ``use_xg`` is True, xG is used as the response variable
    instead of actual goals (with fallback to goals where xG is missing).

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals, match_date,
            and optionally: home_xg, away_xg, home_form, away_form,
            home_squad, away_squad.
        team_index: Mapping from team_id (int) to contiguous integer index.
        decay_rate: Exponential decay factor for temporal weighting.
        use_xg: If True, use xG as response variable.

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

    # Response variable
    if use_xg and "home_xg" in df.columns and "away_xg" in df.columns:
        hg = df["home_xg"].fillna(df["home_goals"]).apply(lambda v: int(round(float(v)))).values
        ag = df["away_xg"].fillna(df["away_goals"]).apply(lambda v: int(round(float(v)))).values
    else:
        hg = df["home_goals"].values.astype(int)
        ag = df["away_goals"].values.astype(int)

    home_goals = hg
    away_goals = ag
    weights = df["weight"].values.astype(float)

    # Covariates (default to neutral if not present)
    home_form = df["home_form"].values.astype(float) if "home_form" in df.columns else np.full(len(df), 0.5)
    away_form = df["away_form"].values.astype(float) if "away_form" in df.columns else np.full(len(df), 0.5)
    home_squad = df["home_squad"].values.astype(float) if "home_squad" in df.columns else np.ones(len(df))
    away_squad = df["away_squad"].values.astype(float) if "away_squad" in df.columns else np.ones(len(df))

    # Centre covariates for better sampling
    home_form_c = home_form - 0.5
    away_form_c = away_form - 0.5
    home_squad_c = home_squad - 1.0
    away_squad_c = away_squad - 1.0

    n_teams = len(team_index)
    n_matches = len(df)

    data_dict = {
        "home_idx": home_idx,
        "away_idx": away_idx,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "weights": weights,
        "home_form": home_form_c,
        "away_form": away_form_c,
        "home_squad": home_squad_c,
        "away_squad": away_squad_c,
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

        squad_coef = pm.Normal("squad_coef", mu=0.0, sigma=0.3)
        form_coef = pm.Normal("form_coef", mu=0.0, sigma=0.3)

        # ── Expected goals ───────────────────────────────────────────────────
        mu_home = pm.math.exp(
            intercept
            + home_advantage
            + attack[home_idx]
            + defense[away_idx]
            + squad_coef * home_squad_c
            + form_coef * home_form_c
        )
        mu_away = pm.math.exp(
            intercept
            + attack[away_idx]
            + defense[home_idx]
            + squad_coef * away_squad_c
            + form_coef * away_form_c
        )

        # ── Poisson log-likelihood ───────────────────────────────────────────
        home_logp = pm.logp(pm.Poisson.dist(mu=mu_home), home_goals)
        away_logp = pm.logp(pm.Poisson.dist(mu=mu_away), away_goals)

        # ── Dixon-Coles correction via pm.Potential ──────────────────────────
        tau_vals = _compute_tau_tensor(
            home_goals, away_goals, mu_home, mu_away, rho
        )

        weighted_ll = weights * (home_logp + away_logp + pm.math.log(tau_vals))
        pm.Potential("weighted_likelihood", weighted_ll.sum())

    logger.info(
        "Built Bayesian model: %d teams, %d matches, covariates=[form, squad].",
        n_teams,
        n_matches,
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
            target_accept=0.9,
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
    """Extract posterior mean and standard deviation for all parameters.

    Args:
        idata: ArviZ InferenceData returned by sample_posterior().
        team_names: Mapping from contiguous integer index → team name string.

    Returns:
        DataFrame with columns: team, attack_mean, attack_std,
        defense_mean, defense_std, home_advantage, squad_coef, form_coef.
    """
    posterior = idata.posterior  # type: ignore[attr-defined]

    attack_mean = posterior["attack"].mean(dim=["chain", "draw"]).values
    attack_std = posterior["attack"].std(dim=["chain", "draw"]).values
    defense_mean = posterior["defense"].mean(dim=["chain", "draw"]).values
    defense_std = posterior["defense"].std(dim=["chain", "draw"]).values
    home_adv = float(
        posterior["home_advantage"].mean(dim=["chain", "draw"]).values
    )

    # Covariate coefficients
    squad_coef_val = float(
        posterior["squad_coef"].mean(dim=["chain", "draw"]).values
    ) if "squad_coef" in posterior else 0.0
    form_coef_val = float(
        posterior["form_coef"].mean(dim=["chain", "draw"]).values
    ) if "form_coef" in posterior else 0.0

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
                "squad_coef": squad_coef_val,
                "form_coef": form_coef_val,
            }
        )

    df = pd.DataFrame(records).sort_values("attack_mean", ascending=False)
    logger.info("Extracted posterior means for %d teams.", len(df))
    return df
