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

if not API_FOOTBALL_KEY:
    logging.warning("API_FOOTBALL_KEY is not set. Data ingestion will fail.")
if not SUPABASE_URL or not SUPABASE_KEY:
    logging.warning("SUPABASE_URL / SUPABASE_KEY not set. Database access will fail.")

# ---------------------------------------------------------------------------
# API-Football
# ---------------------------------------------------------------------------

LEAGUE_ID: int = 71  # Brasileirão Série A
SEASONS: list[int] = [2022, 2023, 2024]  # 2020/2021 not available on current plan

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------

DECAY_RATE: float = 0.92          # Exponential time-decay for form
MAX_GOALS: int = 8                # Poisson grid truncation
MCMC_DRAWS: int = 2000
MCMC_TUNE: int = 1000
MCMC_CHAINS: int = 2
HOME_ADVANTAGE_PRIOR_MEAN: float = 0.3

# ---------------------------------------------------------------------------
# Logging default
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
