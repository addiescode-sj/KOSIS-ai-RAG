"""Layered evaluation (T2-03): L1 retrieval, L2 query understanding, L3 calculation, L4 end to end.

Independent of LangChain (the retrievers and pipelines are only called) and of the Phase 1 eval harness.
Named evals so it does not shadow the Phase 1 `eval` package on the import path.
"""
