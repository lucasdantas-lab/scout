"""Dynamic state-space Dixon-Coles model with time-varying parameters.

Attack and defence parameters follow a Gaussian random walk across rounds,
allowing the model to track teams that improve or decline during a season.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm

from config import (
    DYNAMIC_SIGMA,
    HOME_ADVANTAGE_PRIOR_MEAN,
    MCMC_CHAINS,
    MCMC_DRAWS,
    MCMC_TUNE,
)

logger = logging.getLogger(__name__)


def _extract_round_num(round_str: str) -> int:
    """Extract numeric round number from a round string like 'Regular Season - 15'.

    Args:
        round_str: Round string from API-Football.

    Returns:
        Integer round number, or 0 if parsing fails.
    """
    try:
        return int(str(round_str).split("-")[-1].strip())
    except (ValueError, IndexError):
        return 0


def build_dynamic_model(
    matches_df: pd.DataFrame,
    team_index: dict[int, int],
    sigma: float = DYNAMIC_SIGMA,
) -> tuple[pm.Model, dict[str, Any]]:
    """Construct a dynamic Dixon-Coles model with random-walk parameters.

    Attack and defence parameters evolve per round as::

        attack[t, r] = attack[t, r-1] + ε,  ε ~ Normal(0, σ)
        defense[t, r] = defense[t, r-1] + ε, ε ~ Normal(0, σ)

    Args:
        matches_df: DataFrame of finished matches with columns:
            home_team_id, away_team_id, home_goals, away_goals,
            match_date, round.
        team_index: Mapping from team_id to contiguous integer index.
        sigma: Standard deviation of the random walk innovation.

    Returns:
        Tuple of (pymc.Model, data_dict).
    """
    df = matches_df.dropna(
        subset=["home_goals", "away_goals", "home_team_id", "away_team_id", "round"]
    ).copy()
    df["round_num"] = df["round"].apply(_extract_round_num)
    df = df[df["round_num"] > 0].sort_values("round_num").reset_index(drop=True)

    rounds_sorted = sorted(df["round_num"].unique())
    round_index = {r: i for i, r in enumerate(rounds_sorted)}
    n_rounds = len(rounds_sorted)
    n_teams = len(team_index)

    home_idx = df["home_team_id"].map(team_index).values.astype(int)
    away_idx = df["away_team_id"].map(team_index).values.astype(int)
    round_idx = df["round_num"].map(round_index).values.astype(int)
    home_goals = df["home_goals"].values.astype(int)
    away_goals = df["away_goals"].values.astype(int)

    data_dict = {
        "home_idx": home_idx,
        "away_idx": away_idx,
        "round_idx": round_idx,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "n_teams": n_teams,
        "n_rounds": n_rounds,
        "rounds_sorted": rounds_sorted,
        "round_index": round_index,
    }

    with pm.Model() as model:
        # ── Global priors ──────────────────────────────────────────────────
        home_advantage = pm.Normal(
            "home_advantage", mu=HOME_ADVANTAGE_PRIOR_MEAN, sigma=0.2
        )
        intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)
        sigma_attack = pm.HalfNormal("sigma_attack", sigma=sigma)
        sigma_defense = pm.HalfNormal("sigma_defense", sigma=sigma)

        # ── Random walk priors for attack/defense per round ────────────────
        attack_innovations = pm.Normal(
            "attack_innovations", mu=0.0, sigma=1.0,
            shape=(n_rounds, n_teams),
        )
        defense_innovations = pm.Normal(
            "defense_innovations", mu=0.0, sigma=1.0,
            shape=(n_rounds, n_teams),
        )

        # Cumulative sum to build the random walk
        attack = pm.Deterministic(
            "attack",
            pm.math.cumsum(attack_innovations * sigma_attack, axis=0),
        )
        defense = pm.Deterministic(
            "defense",
            pm.math.cumsum(defense_innovations * sigma_defense, axis=0),
        )

        # ── Expected goals ─────────────────────────────────────────────────
        mu_home = pm.math.exp(
            intercept
            + home_advantage
            + attack[round_idx, home_idx]
            + defense[round_idx, away_idx]
        )
        mu_away = pm.math.exp(
            intercept
            + attack[round_idx, away_idx]
            + defense[round_idx, home_idx]
        )

        # ── Likelihood ─────────────────────────────────────────────────────
        pm.Poisson("home_goals_obs", mu=mu_home, observed=home_goals)
        pm.Poisson("away_goals_obs", mu=mu_away, observed=away_goals)

    logger.info(
        "Built dynamic model: %d teams, %d rounds, %d matches.",
        n_teams, n_rounds, len(df),
    )
    return model, data_dict


def get_current_params(
    idata: az.InferenceData,
    team_names: dict[int, str],
    current_round: int | None = None,
) -> pd.DataFrame:
    """Extract parameters from the latest (or specified) round.

    Args:
        idata: ArviZ InferenceData from the dynamic model.
        team_names: Mapping from integer index → team name.
        current_round: Round index to extract. If None, uses the last round.

    Returns:
        DataFrame with columns: team, attack_mean, attack_std,
        defense_mean, defense_std, home_advantage.
    """
    posterior = idata.posterior  # type: ignore[attr-defined]
    attack = posterior["attack"]
    defense = posterior["defense"]

    n_rounds = attack.shape[2] if len(attack.shape) > 2 else attack.shape[-2]
    r_idx = current_round if current_round is not None else n_rounds - 1

    attack_r = attack[:, :, r_idx, :]  # (chain, draw, team)
    defense_r = defense[:, :, r_idx, :]

    attack_mean = attack_r.mean(dim=["chain", "draw"]).values
    attack_std = attack_r.std(dim=["chain", "draw"]).values
    defense_mean = defense_r.mean(dim=["chain", "draw"]).values
    defense_std = defense_r.std(dim=["chain", "draw"]).values

    home_adv = float(
        posterior["home_advantage"].mean(dim=["chain", "draw"]).values
    )

    records = []
    for idx, name in team_names.items():
        records.append({
            "team": name,
            "attack_mean": float(attack_mean[idx]),
            "attack_std": float(attack_std[idx]),
            "defense_mean": float(defense_mean[idx]),
            "defense_std": float(defense_std[idx]),
            "home_advantage": home_adv,
        })

    df = pd.DataFrame(records).sort_values("attack_mean", ascending=False)
    logger.info(
        "Extracted dynamic params for %d teams at round index %d.", len(df), r_idx
    )
    return df


def plot_param_evolution(
    idata: az.InferenceData,
    team_id: int,
    team_name: str,
    rounds_sorted: list[int] | None = None,
) -> go.Figure:
    """Plot attack and defence parameter evolution across rounds.

    Draws the posterior mean with 50% and 90% credible intervals.

    Args:
        idata: ArviZ InferenceData from the dynamic model.
        team_id: Contiguous integer index of the team.
        team_name: Display name for the team.
        rounds_sorted: List of actual round numbers for x-axis labels.

    Returns:
        Plotly Figure with attack and defence evolution.
    """
    posterior = idata.posterior  # type: ignore[attr-defined]

    attack_samples = posterior["attack"][:, :, :, team_id].values  # (chain, draw, round)
    defense_samples = posterior["defense"][:, :, :, team_id].values

    # Flatten chains
    attack_flat = attack_samples.reshape(-1, attack_samples.shape[-1])  # (samples, rounds)
    defense_flat = defense_samples.reshape(-1, defense_samples.shape[-1])

    n_rounds = attack_flat.shape[1]
    x = rounds_sorted if rounds_sorted and len(rounds_sorted) == n_rounds else list(range(1, n_rounds + 1))

    fig = go.Figure()

    for param_name, samples, color in [
        ("Ataque", attack_flat, "blue"),
        ("Defesa", defense_flat, "red"),
    ]:
        mean = np.mean(samples, axis=0)
        q5, q25, q75, q95 = np.percentile(samples, [5, 25, 75, 95], axis=0)

        # 90% CI
        fig.add_trace(go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(q5) + list(q95)[::-1],
            fill="toself",
            fillcolor=f"rgba({'0,0,255' if color == 'blue' else '255,0,0'}, 0.1)",
            line=dict(width=0),
            name=f"{param_name} 90% CI",
            showlegend=False,
        ))

        # 50% CI
        fig.add_trace(go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(q25) + list(q75)[::-1],
            fill="toself",
            fillcolor=f"rgba({'0,0,255' if color == 'blue' else '255,0,0'}, 0.2)",
            line=dict(width=0),
            name=f"{param_name} 50% CI",
            showlegend=False,
        ))

        # Mean
        fig.add_trace(go.Scatter(
            x=list(x),
            y=list(mean),
            mode="lines",
            name=param_name,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=f"Evolução dos Parâmetros — {team_name}",
        xaxis_title="Rodada",
        yaxis_title="Valor do Parâmetro",
        template="plotly_white",
        height=450,
    )

    return fig
