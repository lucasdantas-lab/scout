"""Entry point for Streamlit Cloud deployment."""

import sys
from pathlib import Path

# Must be first — ensures all internal modules resolve correctly
# regardless of where Streamlit Cloud sets the working directory.
_root = Path(__file__).parent.resolve()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Force dashboard.py to re-execute on every Streamlit rerun.
# Without this, Python's module cache (sys.modules) causes
# `from app.dashboard import *` to skip re-execution, so the
# if/elif page-routing blocks never run after the first load.
if "app.dashboard" in sys.modules:
    del sys.modules["app.dashboard"]

from app.dashboard import *  # noqa: F401, F403, E402
