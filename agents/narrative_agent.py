"""Narrative agent — generates analytical match previews using Claude.

Produces concise, data-driven pre-match narratives in Brazilian Portuguese
based on model predictions, context, and team parameters.
"""

import json
import logging
from typing import Any

from config import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_MATCH_SYSTEM_PROMPT = """Você é um analista de futebol especializado no Brasileirão \
Série A. Gere uma análise pré-jogo concisa e técnica baseada EXCLUSIVAMENTE nos dados \
fornecidos. Sem especulação além dos dados. Sem torcer por nenhum lado. Tom analítico, \
direto. Máximo 4 parágrafos. Escreva em português brasileiro.

Estrutura:
1. Parágrafo de contexto: o que os λ dizem sobre o jogo
2. Parágrafo de mercados: leitura de BTTS, over/under, placar mais provável e o que isso implica
3. Parágrafo de fatores de risco: desfalques, fadiga, incerteza alta de algum parâmetro
4. Uma frase de síntese com o favorito e a ressalva mais importante"""

_ROUND_SYSTEM_PROMPT = """Você é um analista de futebol especializado no Brasileirão \
Série A. Gere um resumo analítico da rodada baseado nos dados fornecidos. \
Tom técnico e direto. Máximo 6 parágrafos. Escreva em português brasileiro.

Destaque:
- Os jogos com maior incerteza do modelo (probabilidades mais equilibradas)
- Onde os ajustes contextuais foram mais impactantes
- O jogo com maior λ combinado e o jogo mais defensivo
- Padrões gerais da rodada"""

_FALLBACK_NARRATIVE = (
    "Análise não disponível. O agente narrativo não conseguiu gerar "
    "a análise para este jogo."
)


def generate_match_narrative(
    prediction_dict: dict[str, Any],
    context_dict: dict[str, Any] | None,
    posterior_means: dict[str, Any] | None,
    home_team: str,
    away_team: str,
) -> str:
    """Generate a match preview narrative using Claude.

    Args:
        prediction_dict: Full prediction output from predict_match(),
            including lambdas, markets, exact_scores, adjustment_log.
        context_dict: Processed context from context_agent.
        posterior_means: Dict with attack/defense means and stds for
            both teams (for uncertainty commentary).
        home_team: Home team name.
        away_team: Away team name.

    Returns:
        Narrative string in Portuguese. Returns fallback text on error.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — returning fallback narrative.")
        return _FALLBACK_NARRATIVE

    try:
        import anthropic

        # Build payload with all relevant data
        payload = {
            "jogo": f"{home_team} x {away_team}",
            "lambda_home_original": round(prediction_dict.get("lambda_home", 0), 3),
            "lambda_away_original": round(prediction_dict.get("lambda_away", 0), 3),
            "lambda_home_ajustado": round(prediction_dict.get("lambda_home_adjusted", 0), 3),
            "lambda_away_ajustado": round(prediction_dict.get("lambda_away_adjusted", 0), 3),
            "prob_1x2": prediction_dict.get("markets_1x2", {}),
            "btts": round(prediction_dict.get("btts", 0), 3),
            "over_under_25": prediction_dict.get("over_under", {}),
            "top_placares": prediction_dict.get("exact_scores", [])[:6],
            "ajustes_aplicados": prediction_dict.get("adjustment_log", []),
        }

        if context_dict:
            payload["contexto"] = {
                "home": {
                    k: context_dict.get("home", {}).get(k)
                    for k in ["ausencias_confirmadas", "duvidas", "lambda_delta", "confianca", "notas"]
                },
                "away": {
                    k: context_dict.get("away", {}).get(k)
                    for k in ["ausencias_confirmadas", "duvidas", "lambda_delta", "confianca", "notas"]
                },
            }

        if posterior_means:
            payload["incerteza_parametros"] = posterior_means

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_MATCH_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            }],
        )

        narrative = response.content[0].text.strip()
        logger.info(
            "generate_match_narrative: generated %d chars for %s x %s.",
            len(narrative),
            home_team,
            away_team,
        )
        return narrative

    except Exception as exc:
        logger.error("Narrative generation failed: %s", exc)
        return _FALLBACK_NARRATIVE


def generate_round_summary(
    predictions_list: list[dict[str, Any]],
    round_num: int | str,
    season: int,
) -> str:
    """Generate an analytical summary of an entire round.

    Args:
        predictions_list: List of prediction dicts (one per match),
            each containing team names, markets, lambdas, adjustment_log.
        round_num: Round number or identifier.
        season: Season year.

    Returns:
        Round summary string in Portuguese. Returns fallback on error.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — returning fallback summary.")
        return "Resumo da rodada não disponível."

    try:
        import anthropic

        # Summarise each match for the payload
        matches_summary = []
        for pred in predictions_list:
            matches_summary.append({
                "jogo": pred.get("match_label", ""),
                "prob_1x2": pred.get("markets_1x2", {}),
                "lambda_home": round(pred.get("lambda_home_adjusted", 0), 3),
                "lambda_away": round(pred.get("lambda_away_adjusted", 0), 3),
                "btts": round(pred.get("btts", 0), 3),
                "ajustes": pred.get("adjustment_log", []),
            })

        payload = {
            "rodada": str(round_num),
            "temporada": season,
            "jogos": matches_summary,
        }

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_ROUND_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            }],
        )

        summary = response.content[0].text.strip()
        logger.info(
            "generate_round_summary: generated %d chars for round %s.",
            len(summary),
            round_num,
        )
        return summary

    except Exception as exc:
        logger.error("Round summary generation failed: %s", exc)
        return "Resumo da rodada não disponível."
