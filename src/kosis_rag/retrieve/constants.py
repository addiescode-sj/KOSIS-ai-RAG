"""Constants of the retrieve stage (T2-41 S2; the question index ones from hq/index.py)."""

from __future__ import annotations

LIST_K = 20  # default tables a dense / BM25 search returns (callers pass their own pool)
HQ_K = 50  # default tables a question index search returns
HQ_N_QUESTIONS = 200  # questions fetched before pooling per table
