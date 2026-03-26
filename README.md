# SCOUT — Statistical Core for Outcome Understanding Tool

Sistema de probabilidade para o Brasileirao Serie A baseado no modelo
Dixon-Coles com inferencia bayesiana via PyMC, enriquecido por agentes
de inteligencia baseados na API do Claude para contexto, narrativa e
autocalibracaco.

---

## Visao Geral

SCOUT combina modelagem estatistica avancada com agentes de IA:

| Componente | Metodo | Uso |
|---|---|---|
| **Dixon-Coles MLE** | `scipy.optimize` L-BFGS-B | Backtest rapido, baseline |
| **Bayesiano** | PyMC NUTS + covariates | Previsoes com incerteza |
| **Dinamico** | Random walk state-space | Evolucao temporal dos parametros |
| **Context Agent** | Claude API + web scraping | Desfalques, lesoes, contexto |
| **Narrative Agent** | Claude API | Analises pre-jogo em portugues |
| **Calibration Agent** | Claude API | Deteccao de padroes de erro |

Mercados derivados: 1X2, BTTS, Over/Under, placares exatos.

---

## Stack

- Python 3.13
- PyMC >= 5.x (NUTS sampling)
- pandas / numpy / scipy
- supabase-py (PostgreSQL via Supabase)
- httpx (ingestao assincrona da API-Football)
- anthropic >= 0.49.0 (Claude API para agentes)
- beautifulsoup4 (scraping de contexto)
- Streamlit + Plotly (dashboard 5 paginas)
- scikit-learn (calibracao)
- arviz (diagnostico MCMC)

---

## Setup

```bash
# 1. Entrar no diretorio
cd scout

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variaveis de ambiente
cp .env.example .env
# Edite .env e preencha:
#   API_FOOTBALL_KEY, SUPABASE_URL, SUPABASE_KEY, ANTHROPIC_API_KEY

# 5. Criar tabelas no Supabase
# Copie o conteudo de data/schema.sql e execute no SQL Editor do Supabase
```

---

## Ingestao de Dados

```bash
python scripts/ingest.py --seasons 2023 2024
```

Ou programaticamente:

```python
import asyncio
from data.ingestion import APIFootballClient
from data.repository import MatchRepository

async def ingest():
    client = APIFootballClient()
    results, teams = await client.bulk_ingest([2023, 2024])
    repo = MatchRepository()
    for t in teams:
        repo.upsert_team(t)
    for entry in results:
        repo.upsert_match(entry["match"])
        if "stats" in entry:
            repo.upsert_stats(entry["stats"])
        if "lineups" in entry:
            for lu in entry["lineups"]:
                repo.upsert_lineup(lu)
        if "events" in entry:
            repo.upsert_events(entry["events"])

asyncio.run(ingest())
```

---

## Treinar o Modelo

### Dixon-Coles MLE (rapido)

```bash
python scripts/train.py --mode mle
```

### Bayesiano (PyMC)

```bash
python scripts/train.py --mode bayesian
```

### Programatico

```python
from data.repository import MatchRepository
from model.bayesian import build_bayesian_model, sample_posterior, get_posterior_means

matches = MatchRepository().get_finished_matches()
team_ids = sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
team_index = {tid: i for i, tid in enumerate(team_ids)}

model, _ = build_bayesian_model(matches, team_index)
idata = sample_posterior(model)  # salva trace em traces/

team_names = {i: str(tid) for i, tid in enumerate(team_ids)}
posterior_df = get_posterior_means(idata, team_names)
```

---

## Gerar Previsoes

```bash
python scripts/predict.py
```

```python
from model.markets import predict_match

result = predict_match(
    home_team_id=123,
    away_team_id=456,
    posterior_means=posterior_df,
    features_dict={"home_form": 0.72, "away_form": 0.55},
    context_json=context_from_agent,
)
print(result["markets_1x2"])
print(result["btts"])
print(result["lambda_home_adjusted"], result["lambda_away_adjusted"])
```

---

## Agentes de Inteligencia

### Context Agent

Coleta noticias de fontes publicas (ge.globo.com) e usa o Claude para
extrair desfalques, duvidas e ajustes de lambda:

```python
from agents.context_agent import get_match_context
context = await get_match_context(match_id, "Flamengo", "Palmeiras", match_date, repo)
# Retorna: {"home": {"ausencias_confirmadas": [...], "lambda_delta": -0.15, ...}, "away": {...}}
```

### Narrative Agent

Gera analises pre-jogo em portugues:

```python
from agents.narrative_agent import generate_match_narrative
narrative = generate_match_narrative(prediction, context, params, "Flamengo", "Palmeiras")
```

### Calibration Agent

Analisa erros apos cada rodada e sugere ajustes:

```python
from agents.calibration_agent import analyze_round_errors, generate_calibration_insights
errors = analyze_round_errors(predictions_df, results_df, round_num=15)
insights = generate_calibration_insights(errors)
```

---

## Rodar o Dashboard

```bash
streamlit run app/dashboard.py
```

Acesse `http://localhost:8501` — 5 paginas:

1. **Rodada** — previsoes por rodada com match cards
2. **Analise de Jogo** — deep-dive com matriz de placares e narrativa
3. **Performance** — metricas, reliability diagram, backtest RPS
4. **Parametros** — rankings, violin plots, evolucao temporal
5. **Agentes** — historico de calibracao, impacto do contexto

---

## Rodar os Testes

```bash
pytest tests/ -v
```

43 testes cobrindo: Dixon-Coles, markets, calibracao, context agent (com mock).

---

## Modos de Operacao

### MODO TREINO (semanal, apos rodada)

```
bulk_ingest -> build_features -> fit_bayesian ->
evaluate_metrics -> calibration_agent -> save_parameters
```

### MODO PREDICAO (pre-rodada)

```
fetch_upcoming -> context_agent -> predict_match ->
narrative_agent -> save_predictions -> dashboard
```

Os agentes Claude sao stateless — cada chamada e independente.
O estado do sistema vive no banco (Supabase).

---

## Estrutura

```
scout/
├── config.py                # Configuracao e constantes
├── data/
│   ├── ingestion.py         # Cliente async API-Football
│   ├── repository.py        # Acesso ao banco (Supabase)
│   └── schema.sql           # DDL PostgreSQL (10 tabelas)
├── features/
│   ├── team_strength.py     # Forca de ataque/defesa (com xG)
│   ├── form.py              # Forma recente com decaimento
│   ├── context.py           # Fadiga, importancia, altitude
│   └── squad.py             # Forca do elenco por partida
├── model/
│   ├── dixon_coles.py       # MLE baseline (com xG)
│   ├── bayesian.py          # Modelo bayesiano (PyMC + covariates)
│   ├── dynamic.py           # Modelo dinamico (random walk)
│   ├── markets.py           # Probabilidades de mercado + contexto
│   └── calibration.py       # Calibracao pos-modelo
├── agents/
│   ├── context_agent.py     # Coleta e processa contexto (Claude)
│   ├── narrative_agent.py   # Gera narrativas (Claude)
│   └── calibration_agent.py # Analisa erros e sugere ajustes (Claude)
├── evaluation/
│   ├── metrics.py           # BS, RPS, Log-Loss, xG vs Goals
│   └── backtest.py          # Walk-forward + comparacao com mercado
├── app/
│   ├── dashboard.py         # Streamlit app (5 paginas)
│   └── components/          # match_card, score_matrix, narrative_panel, metrics_panel
├── scripts/
│   ├── ingest.py            # CLI de ingestao
│   ├── train.py             # CLI de treinamento
│   └── predict.py           # CLI de previsao
└── tests/                   # pytest (43 testes)
```

---

## Limitacoes Conhecidas

- A API-Football tem limites de request (depende do plano)
- xG nem sempre disponivel para todas as temporadas
- Context agent depende de scraping — pode quebrar se layout mudar
- Modelo dinamico requer mais dados para convergir que o estatico
- Agentes Claude requerem ANTHROPIC_API_KEY (custo por chamada)
