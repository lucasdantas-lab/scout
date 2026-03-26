"""Context agent — collects and processes pre-match context using Claude.

Scrapes public news sources for team information (absences, doubts,
tactical changes) and uses the Claude API to extract structured context
that adjusts match predictions.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from config import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_SEARCH_SOURCES = [
    "https://ge.globo.com/busca/?q={query}&order=recent",
]

_NEUTRAL_CONTEXT: dict[str, Any] = {
    "home": {
        "ausencias_confirmadas": [],
        "duvidas": [],
        "confirmados_importantes": [],
        "lambda_delta": 0.0,
        "confianca": 0.0,
        "notas": "Sem informação contextual disponível.",
    },
    "away": {
        "ausencias_confirmadas": [],
        "duvidas": [],
        "confirmados_importantes": [],
        "lambda_delta": 0.0,
        "confianca": 0.0,
        "notas": "Sem informação contextual disponível.",
    },
}

_SYSTEM_PROMPT = """Você é um analista de futebol especializado no Brasileirão Série A. \
Analise o texto fornecido e extraia informações relevantes sobre o estado dos dois times \
para a partida indicada. Responda APENAS com um JSON válido, sem texto adicional, sem \
markdown, sem backticks.

Estrutura obrigatória:
{
  "home": {
    "ausencias_confirmadas": ["lista de nomes"],
    "duvidas": ["lista de nomes"],
    "confirmados_importantes": ["lista de nomes"],
    "lambda_delta": float entre -0.40 e 0.10,
    "confianca": float entre 0.0 e 1.0,
    "notas": "resumo em uma frase"
  },
  "away": {
    "ausencias_confirmadas": ["lista de nomes"],
    "duvidas": ["lista de nomes"],
    "confirmados_importantes": ["lista de nomes"],
    "lambda_delta": float entre -0.40 e 0.10,
    "confianca": float entre 0.0 e 1.0,
    "notas": "resumo em uma frase"
  }
}

Regras para lambda_delta:
- Ausência de centroavante ou meia criativo titular: -0.20 a -0.30
- Ausência de zagueiro titular: afeta defesa adversária, não lambda
- Time em crise interna / briga de vestiário: -0.10 a -0.15
- Retorno de jogador importante: +0.05 a +0.10
- Sem informação relevante: 0.0
- confianca: 1.0 se fonte oficial, 0.5 se especulativo"""


async def collect_raw_context(
    home_team: str,
    away_team: str,
    match_date: str | datetime,
) -> str:
    """Collect text from public news sources about both teams.

    Searches ge.globo.com for recent articles mentioning each team
    within the 72 hours before the match.

    Args:
        home_team: Home team name.
        away_team: Away team name.
        match_date: Match date for recency filtering.

    Returns:
        Concatenated raw text from relevant news articles.
    """
    if isinstance(match_date, str):
        match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
    if match_date.tzinfo is None:
        match_date = match_date.replace(tzinfo=timezone.utc)

    cutoff = match_date - timedelta(hours=72)
    texts: list[str] = []

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for team_name in [home_team, away_team]:
            for source_url in _SEARCH_SOURCES:
                url = source_url.format(query=team_name.replace(" ", "+"))
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        logger.warning(
                            "Context fetch returned %d for %s",
                            resp.status_code,
                            url,
                        )
                        continue

                    soup = BeautifulSoup(resp.text, "html.parser")

                    # Extract article summaries from search results
                    for article in soup.find_all(
                        "div", class_="widget--info__text-container"
                    )[:5]:
                        title_el = article.find(
                            "div", class_="widget--info__title"
                        )
                        desc_el = article.find(
                            "p", class_="widget--info__description"
                        )
                        title = title_el.get_text(strip=True) if title_el else ""
                        desc = desc_el.get_text(strip=True) if desc_el else ""
                        if title or desc:
                            texts.append(f"[{team_name}] {title}\n{desc}")

                except Exception as exc:
                    logger.warning(
                        "Failed to scrape context for '%s' from %s: %s",
                        team_name,
                        url,
                        exc,
                    )

    raw_text = "\n\n---\n\n".join(texts) if texts else ""
    logger.info(
        "collect_raw_context: collected %d chars from %d snippets.",
        len(raw_text),
        len(texts),
    )
    return raw_text


def process_context_with_claude(
    raw_text: str,
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    """Process raw news text with Claude to extract structured context.

    Args:
        raw_text: Concatenated news text from collect_raw_context().
        home_team: Home team name.
        away_team: Away team name.

    Returns:
        Structured context dict with 'home' and 'away' keys.
        Returns neutral context on any error.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — returning neutral context.")
        return _NEUTRAL_CONTEXT

    if not raw_text.strip():
        logger.info("No raw context text — returning neutral context.")
        return _NEUTRAL_CONTEXT

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        user_message = (
            f"Time mandante: {home_team}\n"
            f"Time visitante: {away_team}\n"
            f"Texto:\n{raw_text[:8000]}"  # truncate to stay within limits
        )

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        content = response.content[0].text.strip()
        parsed = json.loads(content)

        # Validate structure
        for side in ("home", "away"):
            if side not in parsed:
                raise ValueError(f"Missing '{side}' key in response.")
            ctx = parsed[side]
            ctx.setdefault("ausencias_confirmadas", [])
            ctx.setdefault("duvidas", [])
            ctx.setdefault("confirmados_importantes", [])
            ctx["lambda_delta"] = float(
                max(-0.40, min(0.10, ctx.get("lambda_delta", 0.0)))
            )
            ctx["confianca"] = float(
                max(0.0, min(1.0, ctx.get("confianca", 0.5)))
            )
            ctx.setdefault("notas", "")

        logger.info(
            "process_context_with_claude: home Δ=%.2f (conf=%.2f), "
            "away Δ=%.2f (conf=%.2f)",
            parsed["home"]["lambda_delta"],
            parsed["home"]["confianca"],
            parsed["away"]["lambda_delta"],
            parsed["away"]["confianca"],
        )
        return parsed

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Claude JSON response: %s", exc)
        return _NEUTRAL_CONTEXT
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return _NEUTRAL_CONTEXT


async def get_match_context(
    match_id: int,
    home_team: str,
    away_team: str,
    match_date: str | datetime,
    repository: Any,
) -> dict[str, Any]:
    """Get or generate context for a match.

    Checks the database for existing context first. If none exists,
    runs the full collect → process → save pipeline.

    Args:
        match_id: The fixture id.
        home_team: Home team name.
        away_team: Away team name.
        match_date: Match date.
        repository: ModelRepository instance for persistence.

    Returns:
        Processed context dict with 'home' and 'away' keys.
    """
    # Check for existing context
    existing = repository.get_match_context(match_id)
    if existing and existing.get("processed_context"):
        logger.info("Using existing context for match %d.", match_id)
        ctx = existing["processed_context"]
        if isinstance(ctx, str):
            ctx = json.loads(ctx)
        return ctx

    # Collect and process
    raw_text = await collect_raw_context(home_team, away_team, match_date)
    processed = process_context_with_claude(raw_text, home_team, away_team)

    # Persist
    try:
        repository.save_context(
            match_id=match_id,
            raw_news=raw_text[:50000],  # limit storage size
            processed_context=processed,
            model=CLAUDE_MODEL,
        )
    except Exception as exc:
        logger.error("Failed to save context for match %d: %s", match_id, exc)

    return processed
