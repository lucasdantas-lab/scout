"""Script de ingestão de dados da API-Football para o Supabase.

Uso:
    python scripts/ingest.py                        # temporadas padrão (config.py)
    python scripts/ingest.py --seasons 2023 2024    # temporadas específicas
    python scripts/ingest.py --upcoming             # apenas próximos jogos
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Permite rodar a partir da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SEASONS
from data.ingestion import APIFootballClient
from data.repository import MatchRepository

logger = logging.getLogger(__name__)


async def ingest_historical(seasons: list[int]) -> None:
    """Ingere jogos finalizados e estatísticas para as temporadas informadas."""
    client = APIFootballClient()
    repo = MatchRepository()

    logger.info("Iniciando ingestão histórica para temporadas: %s", seasons)
    records, teams = await client.bulk_ingest(seasons)

    # 1. Upsert times primeiro (FK constraint)
    teams_ok = 0
    for team in teams:
        try:
            repo.upsert_team(team)
            teams_ok += 1
        except Exception as exc:
            logger.warning("Falha ao upsert team_id=%s: %s", team.get("id"), exc)
    logger.info("Times salvos: %d", teams_ok)

    # 2. Upsert partidas e estatísticas
    upserted = 0
    errors = 0
    for entry in records:
        try:
            repo.upsert_match(entry["match"])
            if "stats" in entry:
                repo.upsert_stats(entry["stats"])
            upserted += 1
        except Exception as exc:
            errors += 1
            logger.warning(
                "Falha ao upsert match_id=%s: %s",
                entry["match"].get("id"),
                exc,
            )

    logger.info(
        "Ingestão histórica concluída. Partidas: %d | Erros: %d",
        upserted,
        errors,
    )


async def ingest_upcoming() -> None:
    """Busca e salva os próximos 10 jogos agendados."""
    client = APIFootballClient()
    repo = MatchRepository()

    logger.info("Buscando próximos jogos …")
    fixtures, teams = await client.fetch_upcoming(next_n=10)

    for team in teams:
        try:
            repo.upsert_team(team)
        except Exception as exc:
            logger.warning("Falha ao upsert team_id=%s: %s", team.get("id"), exc)

    for match in fixtures:
        try:
            repo.upsert_match(match)
        except Exception as exc:
            logger.warning("Falha ao upsert match_id=%s: %s", match.get("id"), exc)

    logger.info("%d próximos jogos salvos.", len(fixtures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCOUT — ingestão de dados")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=SEASONS,
        metavar="ANO",
        help=f"Temporadas a ingerir (padrão: {SEASONS})",
    )
    parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Busca apenas os próximos jogos agendados (ignora --seasons)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.upcoming:
        asyncio.run(ingest_upcoming())
    else:
        asyncio.run(ingest_historical(args.seasons))
