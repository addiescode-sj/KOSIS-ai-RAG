"""Phase 2 table search RAG: loader -> retriever -> chain -> LangChain tools."""

import os
from pathlib import Path

# Phase 1 (vendor/phase1) reads data/, eval/golden, eval/results and .env under KOSIS_WORKSPACE; default it to this
# repo so nobody has to set it by hand (docs/ISSUES.md P2-07). A value already in the environment wins.
os.environ.setdefault("KOSIS_WORKSPACE", str(Path(__file__).resolve().parents[2]))
