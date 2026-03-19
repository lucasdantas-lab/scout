"""Gera previsões para partidas históricas usando o modelo Dixon-Coles MLE.

Uso:
    python scripts/predict.py                # todas as temporadas
    python scripts/predict.py --season 2024  # só 2024
    python scripts/predict.py --limit 50     # primeiras 50 partidas (teste)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repository import MatchRepository, ModelRepository
from model.dixon_coles import fit_dixon_coles_mle
from model.markets import (
    compute_btts,
    compute_over_under,
    compute_score_matrix,
    compute_1x2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_predictions(season: int | None, limit: int | None) -> None:
    match_repo = MatchRepository()
    model_repo = ModelRepository()

    # Carrega partidas finalizadas
    matches = match_repo.get_finished_matches(season=season)
    if matches.empty:
        logger.error("Sem partidas finalizadas. Rode a ingestão primeiro.")
        return

    logger.info("Ajustando modelo MLE em %d partidas …", len(matches))
    result = fit_dixon_coles_mle(matches)
    if not result["success"]:
        logger.warning("Otimização não convergiu: %s", result["message"])

    team_params = result["team_params"]
    home_adv = result["home_advantage"]
    rho = result["rho"]
    intercept = result.get("intercept", 0.0)

    logger.info(
        "Parâmetros globais: home_advantage=%.4f  rho=%.4f  intercept=%.4f",
        home_adv, rho, intercept,
    )

    if limit:
        matches = matches.head(limit)

    logger.info("Gerando previsões para %d partidas …", len(matches))
    saved = skipped = 0

    for _, row in matches.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])

        if home_id not in team_params or away_id not in team_params:
            logger.debug("Time %d ou %d sem parâmetros, ignorando.", home_id, away_id)
            skipped += 1
            continue

        hp = team_params[home_id]
        ap = team_params[away_id]

        lambda_home = float(np.exp(intercept + home_adv + hp["attack"] + ap["defense"]))
        lambda_away = float(np.exp(intercept + ap["attack"] + hp["defense"]))

        score_mat = compute_score_matrix(lambda_home, lambda_away, rho)
        probs = compute_1x2(score_mat)
        btts = compute_btts(score_mat)
        ou = compute_over_under(score_mat)

        prediction = {
            "match_id": int(row["id"]),
            "model_version": "mle-v1",
            "prob_home": round(probs["home"], 6),
            "prob_draw": round(probs["draw"], 6),
            "prob_away": round(probs["away"], 6),
            "prob_btts": round(btts, 6),
            "prob_over25": round(ou["over"], 6),
            "score_matrix": score_mat.tolist(),
            "lambda_home": round(lambda_home, 6),
            "lambda_away": round(lambda_away, 6),
        }

        try:
            model_repo.save_prediction(prediction)
            saved += 1
            if saved % 100 == 0:
                logger.info("  %d/%d salvas …", saved, len(matches))
        except Exception as exc:
            logger.warning("Erro ao salvar match_id=%s: %s", row["id"], exc)
            skipped += 1

    logger.info("Concluído: %d salvas, %d ignoradas.", saved, skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCOUT — geração de previsões históricas")
    parser.add_argument("--season", type=int, default=None, metavar="ANO")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Limitar a N partidas (útil para testes)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_predictions(season=args.season, limit=args.limit)
