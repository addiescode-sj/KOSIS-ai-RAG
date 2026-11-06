"""Rerank stage step functions (T2-41 S3: the bodies of graph.py's rerank1 and rerank2 nodes).

`resources` is the search resources (pipeline.SearchResources until S4): stage1_score and settings.rerank_n /
rerank_mode for rerank1; select, stage2_name, stage2_llm and settings.stage2_k for rerank2.
"""

from __future__ import annotations

from typing import Any

from kosis_rag.llm.config import require_key
from kosis_rag.rerank.schemas import Stage1Input, Stage1Output, Stage2Input, Stage2Output
from kosis_rag.rerank.stage1 import rerank_stage1
from kosis_rag.rerank.stage2 import rerank_stage2


def rerank1(inp: Stage1Input, resources: Any) -> Stage1Output:
    """Re-order the top rerank_n by the stage 1 cross-encoder.

    In: inp.question, inp.docs. Out: Stage1Output(docs re-ordered with stage1_* metadata).
    Side effects: cross-encoder inference; ValueError for an unknown RERANK_MODE. Node: rerank1.
    """
    s = resources.settings
    return Stage1Output(docs=rerank_stage1(inp.question, inp.docs, resources.stage1_score, n=s.rerank_n,
                                           mode=s.rerank_mode))


def rerank2(inp: Stage2Input, resources: Any) -> Stage2Output:
    """Let the stage 2 LLM pick among the top stage2_k.

    In: inp.question, inp.docs. Out: Stage2Output(docs: picked first, every doc with stage2_* metadata).
    Side effects: raises LLMKeyMissing on entry when resources.stage2_llm (the config the selector was built from)
        has no api_key and its provider is not mock — before any call; a selector given directly (stage2_llm None,
        tests) is not checked. Then one LLM re-ranker call (CLOVA / HCX / OpenAI); its errors are stored in
        stage2_error. Node: rerank2.
    """
    require_key(getattr(resources, "stage2_llm", None))
    docs = rerank_stage2(inp.question, inp.docs, resources.select, k_in=resources.settings.stage2_k,
                         name=resources.stage2_name)
    return Stage2Output(docs=docs)
