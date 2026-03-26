"""Central configuration module for SCOUT.

Reads environment variables and exposes project-wide constants.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment — local (.env) or Streamlit Cloud (st.secrets)
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")


def _get_secret(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) or fall back to env var."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, ""))
    except Exception:
        return os.environ.get(key, "")


API_FOOTBALL_KEY: str = _get_secret("API_FOOTBALL_KEY")
SUPABASE_URL: str = _get_secret("SUPABASE_URL")
SUPABASE_KEY: str = _get_secret("SUPABASE_KEY")
ANTHROPIC_API_KEY: str = _get_secret("ANTHROPIC_API_KEY")

if not API_FOOTBALL_KEY:
    logging.warning("API_FOOTBALL_KEY is not set. Data ingestion will fail.")
if not SUPABASE_URL or not SUPABASE_KEY:
    logging.warning("SUPABASE_URL / SUPABASE_KEY not set. Database access will fail.")
if not ANTHROPIC_API_KEY:
    logging.warning("ANTHROPIC_API_KEY is not set. Agent features will be disabled.")

# ---------------------------------------------------------------------------
# API-Football
# ---------------------------------------------------------------------------

LEAGUE_ID: int = 71  # Brasileirão Série A
SEASONS: list[int] = [2019, 2020, 2021, 2022, 2023, 2024]

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------

DECAY_RATE: float = 0.92          # Exponential time-decay for form
MAX_GOALS: int = 8                # Poisson grid truncation
MCMC_DRAWS: int = 2000
MCMC_TUNE: int = 1000
MCMC_CHAINS: int = 2
HOME_ADVANTAGE_PRIOR_MEAN: float = 0.3
DYNAMIC_SIGMA: float = 0.05      # Volatility for dynamic state-space params

# ---------------------------------------------------------------------------
# Squad strength weights
# ---------------------------------------------------------------------------

SQUAD_WEIGHT_STARTER: float = 1.0
SQUAD_WEIGHT_SUB: float = 0.4

# ---------------------------------------------------------------------------
# Altitude factors (penalize low-altitude visitors)
# ---------------------------------------------------------------------------

ALTITUDE_FACTOR: dict[str, float] = {
    "Cuiabá": 1.08,
    "Brasília": 1.06,
    "Goiânia": 1.05,
    "default": 1.0,
}

# ---------------------------------------------------------------------------
# Claude API (agents)
# ---------------------------------------------------------------------------

CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Logging default
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
