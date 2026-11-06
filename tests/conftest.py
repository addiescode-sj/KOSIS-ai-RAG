"""Phase 2 tests run with KOSIS_WORKSPACE at this repo (Phase 1 reads data/, eval/golden and .env under it).

Importing kosis_rag no longer sets it (T2-41 S1, P2-D5), so the tests do; a value already in the environment wins.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KOSIS_WORKSPACE", str(ROOT))
