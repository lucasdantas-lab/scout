"""Entry point for Streamlit Cloud deployment."""

import sys
from pathlib import Path

# Must be first — ensures all internal modules resolve correctly
# regardless of where Streamlit Cloud sets the working directory.
_root = Path(__file__).parent.resolve()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app.dashboard import *  # noqa: F401, F403, E402
