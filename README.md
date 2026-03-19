# SCOUT — Statistical Core for Outcome Understanding Tool

Sistema de probabilidade para o Brasileirão Série A baseado no modelo
Dixon-Coles com inferência bayesiana via PyMC.

---

## Visão Geral

SCOUT combina um modelo estatístico de gols de Poisson com correção de
dependência Dixon-Coles em duas variantes:

| Variante | Método | Uso |
|---|---|---|
| **MLE** | `scipy.optimize` L-BFGS-B | Backtest rápido, baseline |
| **Bayesiana** | PyMC NUTS | Previsões com incerteza |

Mercados derivados automaticamente: 1X2, BTTS, Over/Under, placares exatos.

---

## Stack

- Python 3.13
- PyMC ≥ 5.x (NUTS sampling)
- pandas / numpy / scipy
- supabase-py (PostgreSQL via Supabase)
- httpx (ingestão assíncrona da API-Football)
- Streamlit + Plotly (dashboard)
- scikit-learn (calibração)

---

## Setup

```bash
# 1. Clonar / entrar no diretório
cd scout

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Edite .env e preencha API_FOOTBALL_KEY, SUPABASE_URL, SUPABASE_KEY

# 5. Criar tabelas no Supabase
# Copie o conteúdo de data/schema.sql e execute no SQL Editor do Supabase
```

---

## Ingestão de Dados

```python
# Executar em modo interativo ou como script
import asyncio
from data.ingestion import APIFootballClient
from data.repository import MatchRepository
from config import SEASONS

async def ingest():
    client = APIFootballClient()
    repo = MatchRepository()

    records = await client.bulk_ingest(SEASONS)
    for entry in records:
        repo.upsert_match(entry["match"])
        if "stats" in entry:
            repo.upsert_stats(entry["stats"])

asyncio.run(ingest())
```

---

## Treinar o Modelo

### Dixon-Coles MLE (rápido)

```python
from data.repository import MatchRepository
from model.dixon_coles import fit_dixon_coles_mle

matches = MatchRepository().get_finished_matches()
params = fit_dixon_coles_mle(matches)
print(params["home_advantage"], params["rho"])
```

### Bayesiano (PyMC)

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
print(posterior_df.head())
```

---

## Gerar Previsões

```python
from model.markets import predict_match

result = predict_match(
    home_team_id=123,
    away_team_id=456,
    posterior_means=posterior_df,
    home_form=0.72,
    away_form=0.55,
)
print(result["markets_1x2"])
print(result["btts"])
print(result["over_under"])
```

---

## Rodar o Dashboard

```bash
streamlit run app/dashboard.py
```

Acesse `http://localhost:8501` no browser.

---

## Rodar os Testes

```bash
pytest tests/ -v
```

---

## Estrutura

```
scout/
├── config.py              # Configuração e constantes
├── data/
│   ├── ingestion.py       # Cliente async API-Football
│   ├── repository.py      # Acesso ao banco (Supabase)
│   └── schema.sql         # DDL PostgreSQL
├── features/
│   ├── team_strength.py   # Força de ataque/defesa
│   ├── form.py            # Forma recente com decaimento
│   └── context.py         # Fadiga, importância, altitude
├── model/
│   ├── dixon_coles.py     # MLE baseline
│   ├── bayesian.py        # Modelo bayesiano (PyMC)
│   ├── markets.py         # Probabilidades de mercado
│   └── calibration.py     # Calibração pós-modelo
├── evaluation/
│   ├── metrics.py         # BS, RPS, Log-Loss
│   └── backtest.py        # Walk-forward validation
├── app/
│   ├── dashboard.py       # Streamlit app (4 páginas)
│   └── components/        # Componentes reutilizáveis
└── tests/                 # pytest
```
