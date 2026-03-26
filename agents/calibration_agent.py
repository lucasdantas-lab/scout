"""Calibration agent — analyses model errors and suggests adjustments using Claude.

After each completed round, this agent examines prediction errors, identifies
systematic patterns, and generates actionable calibration insights.
"""

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from config import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Você é um estatístico especializado em calibração de modelos \
preditivos para futebol. Analise os padrões de erro fornecidos e sugira ajustes \
específicos e acionáveis para melhorar o modelo. Seja preciso e quantitativo. \
Responda em JSON com a seguinte estrutura:
{
  "padroes_identificados": ["lista de padrões"],
  "ajustes_sugeridos": [
    {
      "parametro": "nome do parâmetro",
      "ajuste": "descrição do ajuste",
      "magnitude": float,
      "confianca": float,
      "justificativa": "explicação técnica"
    }
  ],
  "alertas": ["problemas que precisam de atenção"],
  "resumo": "análise em 2-3 frases"
}"""

_NEUTRAL_INSIGHTS: dict[str, Any] = {
    "padroes_identificados": [],
    "ajustes_sugeridos": [],
    "alertas": [],
    "resumo": "Análise de calibração não disponível.",
}


def analyze_round_errors(
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
    round_num: int | str,
) -> dict[str, Any]:
    """Calculate per-match error metrics for a completed round.

    Identifies systematic patterns in prediction errors:
    - Midweek (Wednesday) games
    - Post-FIFA-date rounds
    - Relegation zone teams
    - Home advantage over/under-estimation

    Args:
        predictions_df: Predictions for the round with columns:
            match_id, prob_home, prob_draw, prob_away, lambda_home,
            lambda_away.
        results_df: Actual results with columns: match_id, home_goals,
            away_goals, match_date, home_team_id, away_team_id.
        round_num: Round identifier.

    Returns:
        Structured error summary dict.
    """
    merged = predictions_df.merge(
        results_df[["match_id", "home_goals", "away_goals", "match_date"]],
        on="match_id",
        how="inner",
        suffixes=("_pred", "_actual"),
    )

    if merged.empty:
        logger.warning("No matched predictions for round %s.", round_num)
        return {"round": str(round_num), "n_matches": 0, "errors": []}

    errors: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        # Actual outcome
        if hg > ag:
            actual = "home"
        elif hg < ag:
            actual = "away"
        else:
            actual = "draw"

        predicted_prob = float(row.get(f"prob_{actual}", 0.33))
        lambda_h = float(row.get("lambda_home", 1.0))
        lambda_a = float(row.get("lambda_away", 1.0))

        # Squared error per outcome
        brier = sum(
            (float(row.get(f"prob_{o}", 0.33)) - (1.0 if o == actual else 0.0)) ** 2
            for o in ("home", "draw", "away")
        )

        # Day of week analysis
        match_date = pd.to_datetime(row.get("match_date"))
        is_midweek = match_date.weekday() in (1, 2, 3) if match_date else False

        errors.append({
            "match_id": int(row["match_id"]),
            "actual_result": actual,
            "predicted_prob": round(predicted_prob, 3),
            "brier_score": round(brier, 4),
            "lambda_home": round(lambda_h, 3),
            "lambda_away": round(lambda_a, 3),
            "actual_goals": f"{hg}-{ag}",
            "goal_diff_error": abs((lambda_h - lambda_a) - (hg - ag)),
            "is_midweek": is_midweek,
            "total_goals_error": abs((lambda_h + lambda_a) - (hg + ag)),
        })

    # Aggregate patterns
    errors_arr = np.array([e["brier_score"] for e in errors])
    midweek_errors = [e["brier_score"] for e in errors if e["is_midweek"]]
    weekend_errors = [e["brier_score"] for e in errors if not e["is_midweek"]]

    home_overest = sum(
        1 for e in errors
        if e["actual_result"] != "home"
        and float(merged[merged["match_id"] == e["match_id"]]["prob_home"].iloc[0]) > 0.5
    )

    summary = {
        "round": str(round_num),
        "n_matches": len(errors),
        "mean_brier": round(float(errors_arr.mean()), 4),
        "std_brier": round(float(errors_arr.std()), 4),
        "midweek_mean_brier": round(
            float(np.mean(midweek_errors)), 4
        ) if midweek_errors else None,
        "weekend_mean_brier": round(
            float(np.mean(weekend_errors)), 4
        ) if weekend_errors else None,
        "home_overestimation_count": home_overest,
        "mean_total_goals_error": round(
            float(np.mean([e["total_goals_error"] for e in errors])), 3
        ),
        "errors": errors,
    }

    logger.info(
        "analyze_round_errors: round=%s, n=%d, mean_brier=%.4f",
        round_num,
        len(errors),
        summary["mean_brier"],
    )
    return summary


def generate_calibration_insights(
    error_summary: dict[str, Any],
    historical_patterns: list[dict[str, Any]] | None = None,
    round_num: int | str = "",
) -> dict[str, Any]:
    """Use Claude to analyse error patterns and suggest adjustments.

    Args:
        error_summary: Output from analyze_round_errors().
        historical_patterns: Optional list of previous error summaries
            for trend detection.
        round_num: Round identifier for logging.

    Returns:
        Structured insights dict. Returns neutral insights on error.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — returning neutral insights.")
        return _NEUTRAL_INSIGHTS

    try:
        import anthropic

        payload = {
            "rodada_atual": error_summary,
            "historico": historical_patterns[:5] if historical_patterns else [],
        }

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            }],
        )

        content = response.content[0].text.strip()
        parsed = json.loads(content)

        # Validate structure
        parsed.setdefault("padroes_identificados", [])
        parsed.setdefault("ajustes_sugeridos", [])
        parsed.setdefault("alertas", [])
        parsed.setdefault("resumo", "")

        logger.info(
            "generate_calibration_insights: round=%s, %d patterns, %d adjustments.",
            round_num,
            len(parsed["padroes_identificados"]),
            len(parsed["ajustes_sugeridos"]),
        )
        return parsed

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse calibration JSON response: %s", exc)
        return _NEUTRAL_INSIGHTS
    except Exception as exc:
        logger.error("Calibration agent API call failed: %s", exc)
        return _NEUTRAL_INSIGHTS


def save_and_return_insights(
    error_summary: dict[str, Any],
    insights: dict[str, Any],
    round_num: int | str,
    season: int,
    repository: Any,
) -> dict[str, Any]:
    """Save calibration insights to database and return them.

    Args:
        error_summary: Error analysis from analyze_round_errors().
        insights: Insights from generate_calibration_insights().
        round_num: Round identifier.
        season: Season year.
        repository: ModelRepository instance.

    Returns:
        The insights dict.
    """
    log_record = {
        "round_analyzed": str(round_num),
        "season": season,
        "error_patterns": error_summary,
        "suggested_adjustments": insights.get("ajustes_sugeridos", []),
        "agent_reasoning": insights.get("resumo", ""),
        "applied": False,
    }

    try:
        repository.save_calibration_log(log_record)
        logger.info(
            "Saved calibration log for round %s, season %d.", round_num, season
        )
    except Exception as exc:
        logger.error("Failed to save calibration log: %s", exc)

    return insights


def apply_approved_adjustments(
    calibration_id: int,
    repository: Any,
) -> dict[str, Any]:
    """Mark a calibration adjustment as applied.

    Args:
        calibration_id: ID of the calibration_log entry.
        repository: ModelRepository instance (must have _client attribute).

    Returns:
        Dict with the adjustments that were applied.
    """
    try:
        response = (
            repository._client.table("calibration_log")
            .select("*")
            .eq("id", calibration_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            logger.warning("Calibration log %d not found.", calibration_id)
            return {}

        record = response.data[0]

        # Mark as applied
        repository._client.table("calibration_log").update(
            {"applied": True}
        ).eq("id", calibration_id).execute()

        logger.info("Marked calibration log %d as applied.", calibration_id)
        return {
            "id": calibration_id,
            "adjustments": record.get("suggested_adjustments", {}),
            "reasoning": record.get("agent_reasoning", ""),
        }

    except Exception as exc:
        logger.error(
            "Failed to apply calibration adjustments for id %d: %s",
            calibration_id,
            exc,
        )
        return {}
