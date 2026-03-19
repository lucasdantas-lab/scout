"""Script de treinamento do modelo Dixon-Coles (MLE e Bayesiano).

Uso:
    python scripts/train.py --mode mle                # apenas MLE (rápido)
    python scripts/train.py --mode bayesian           # apenas Bayesiano (MCMC)
    python scripts/train.py --mode both               # ambos (padrão)
    python scripts/train.py --mode mle --season 2024  # filtra por temporada
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MCMC_CHAINS, MCMC_DRAWS, MCMC_TUNE
from data.repository import MatchRepository, ModelRepository
from model.dixon_coles import fit_dixon_coles_mle
from model.bayesian import build_bayesian_model, sample_posterior, get_posterior_means

logger = logging.getLogger(__name__)


def run_mle(season: int | None) -> None:
    """Ajusta o modelo MLE e salva os parâmetros no banco."""
    match_repo = MatchRepository()
    model_repo = ModelRepository()

    matches = match_repo.get_finished_matches(season=season)
    if matches.empty:
        logger.error("Sem partidas finalizadas. Rode a ingestão primeiro.")
        return

    logger.info("Ajustando Dixon-Coles MLE em %d partidas …", len(matches))
    result = fit_dixon_coles_mle(matches)

    if not result["success"]:
        logger.warning("Otimização não convergiu: %s", result["message"])

    used_season = season or int(matches["season"].max())
    for team_id, params in result["team_params"].items():
        model_repo.save_parameters(
            run_id="mle",
            team_id=team_id,
            attack=params["attack"],
            defense=params["defense"],
            season=used_season,
            parameter_type="posterior_mean",
        )

    logger.info(
        "MLE: home_advantage=%.4f  rho=%.4f  — %d times salvos.",
        result["home_advantage"],
        result["rho"],
        len(result["team_params"]),
    )


def run_bayesian(season: int | None) -> None:
    """Ajusta o modelo Bayesiano (MCMC) e salva os parâmetros no banco."""
    match_repo = MatchRepository()
    model_repo = ModelRepository()

    matches = match_repo.get_finished_matches(season=season)
    if matches.empty:
        logger.error("Sem partidas finalizadas. Rode a ingestão primeiro.")
        return

    team_ids = sorted(
        set(matches["home_team_id"]).union(matches["away_team_id"])
    )
    team_index = {tid: i for i, tid in enumerate(team_ids)}

    logger.info(
        "Construindo modelo Bayesiano: %d times, %d partidas …",
        len(team_ids),
        len(matches),
    )
    model, _ = build_bayesian_model(matches, team_index)

    logger.info(
        "Iniciando NUTS: draws=%d  tune=%d  chains=%d",
        MCMC_DRAWS, MCMC_TUNE, MCMC_CHAINS,
    )
    idata = sample_posterior(model)

    # Nomes como string do team_id (substitua por nomes reais se disponível)
    team_names = {i: str(tid) for i, tid in enumerate(team_ids)}
    posterior_df = get_posterior_means(idata, team_names)

    used_season = season or int(matches["season"].max())
    for _, row in posterior_df.iterrows():
        # Recupera team_id a partir do nome (que é str(team_id))
        try:
            team_id = int(row["team"])
        except ValueError:
            continue

        model_repo.save_parameters(
            run_id="bayesian",
            team_id=team_id,
            attack=row["attack_mean"],
            defense=row["defense_mean"],
            season=used_season,
            parameter_type="posterior_mean",
        )
        model_repo.save_parameters(
            run_id="bayesian",
            team_id=team_id,
            attack=row["attack_std"],
            defense=row["defense_std"],
            season=used_season,
            parameter_type="posterior_std",
        )

    logger.info("Bayesiano: %d times salvos.", len(posterior_df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCOUT — treinamento do modelo")
    parser.add_argument(
        "--mode",
        choices=["mle", "bayesian", "both"],
        default="both",
        help="Qual modelo treinar (padrão: both)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        metavar="ANO",
        help="Filtrar partidas por temporada (padrão: todas)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in ("mle", "both"):
        run_mle(args.season)

    if args.mode in ("bayesian", "both"):
        run_bayesian(args.season)
