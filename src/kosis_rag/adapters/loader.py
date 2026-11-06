"""Open every heavy part of the staged search once (T2-41 S4: pipeline.load_resources, now over IndexSpec and
SearchOptions instead of SearchSettings).

    question -> [T2-14 conditions] -> fine-tuned bge-m3 (Chroma) top 50 + BM25 top 50 [+ question index] -> RRF
             -> stage 1: fine-tuned bge-reranker-v2-m3 over the top 50 (when spec.rerank_ckpt is set)
             -> stage 2: LLM re-ranker over the top 20 (options.stage2 = clova | hcx | gpt | none)

Nothing here reads .env or the environment (§0.6): the caller declares the spec, the options and the stage 2 LLM
(backend.app.config.search_config_from_env reads the shell environment for the backend and the scripts). BM25, the
re-ranker and the collection use one table text format (shared.tabletext.text_format): the collection's own
doc_format, which spec.text_format (when set) must match; weights trained on another format are refused
(adapters.models).

The question index (T2-39) is read by the graph only (graph.search adds retrieve_hq / retrieve_hq_bm25);
legacy.pipeline.build_search, the nested-retriever path, ignores it.
"""

from __future__ import annotations

import logging
from typing import Any

from kosis_rag.adapters import files
from kosis_rag.legacy.retrievers import BM25TableRetriever, ChromaTableRetriever, HqRetriever
from kosis_rag.llm.config import LLMConfig
from kosis_rag.rerank import selectors
from kosis_rag.rerank.stage1 import passage_scorer
from kosis_rag.rerank.stage2 import Score, Select
from kosis_rag.retrieve import bm25 as table_bm25
from kosis_rag.retrieve import hq as question_index
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.schemas.resources import OFF, IndexSpec, SearchOptions, SearchResources
from kosis_rag.shared import tabletext as text

log = logging.getLogger(__name__)
_SIGUNGU_WARNED: list[bool] = []  # set by the first load_resources that finds no sigungu.json


def load_resources(*, spec: IndexSpec, options: SearchOptions, llm: LLMConfig | None = None,
                   records: list[TableRecord] | None = None, dense: Any | None = None, bm25: Any | None = None,
                   stage1_score: Score | None = None, select: Select | None = None, hq: Any | None = None,
                   hq_bm25: Any | None = None, sigungu: dict[str, list[str]] | None = None) -> SearchResources:
    """Load (or take the given fakes for) every heavy part of the staged search once.

    In (keyword only, so spec and options cannot be swapped — search_config_from_env returns options first):
        spec = paths, Chroma and checkpoints; options = how the search behaves; llm = the stage 2 LLM's config
        (None: selectors.default_config(options.stage2), no key — rerank2 raises LLMKeyMissing on entry);
        records, dense, bm25, stage1_score, select, hq, hq_bm25, sigungu = parts that replace the loaded ones
        (tests pass fakes here so nothing real is touched).
    Out: SearchResources. stage1_score is set when spec.rerank_ckpt is set or a scorer is given; select when
        options.stage2 names a known re-ranker or a selector is given. stage2_name = options.stage2, or "custom" for a
        given selector with stage 2 off. BM25 returns every table when the condition filter is on (it is filtered
        after the search), else `pool`. hq / hq_bm25 (T2-39) are set only when spec.hq_build is set (hq_bm25 also
        needs options.hq_bm25); otherwise None, even when given. sigungu = the 시군구 -> 시도 dictionary parse reads
        (spec.sigungu or data/sigungu.json, {} when the file is missing). stage2_llm = the config the selector is
        built from (`llm`, else the default config without a key); with a given selector, `llm` (default None:
        rerank2 checks no key); None when stage 2 is off.
    Side effects: reads tables.jsonl and sigungu.json (one warning when it is missing), connects to Chroma and loads
        the query embedder, loads the cross-encoder, opens the question collection (reusing the dense retriever's
        query embedder) and reads its questions.jsonl, unless the matching part is given. Raises ValueError for a
        spec.text_format that disagrees with the collection or an unknown options.stage2, as before.
    Node: none (called once before build_graph / build_search / SearchModule).
    """
    o = options
    # 1. registry, dense collection and the text format they agree on
    records = records if records is not None else files.read_records(spec.tables or files.TABLES)
    if dense is None:
        dense = ChromaTableRetriever.connect(spec.chroma.collection, k=o.pool, device=spec.device,
                                             embed_ckpt=spec.embed_ckpt, chroma=spec.chroma)
    fmt = text.text_format(spec, dense)
    if bm25 is None:
        bm25 = BM25TableRetriever.from_index(  # filtered after: score all
            table_bm25.bm25_text(records, k=len(records) if o.cond_filter else o.pool, fmt=fmt))

    # 2. stage 1 scorer (fine-tuned cross-encoder)
    text_of: dict[str, str] | None = None
    if spec.rerank_ckpt and stage1_score is None:
        from kosis_rag.adapters.models import load_cross_encoder

        text_of = {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
        scorer = load_cross_encoder(spec.rerank_ckpt, device=spec.device, doc_format=fmt)
        stage1_score = passage_scorer(scorer, text_of, text.norm_query)

    # 3. stage 2 selector (LLM re-ranker)
    stage2_llm = llm
    if select is None and o.stage2 not in OFF:
        if o.stage2 not in selectors.LLM_RERANKERS:
            raise ValueError(f"unknown STAGE2 {o.stage2!r}; known: none, {', '.join(selectors.LLM_RERANKERS)}")
        text_of = text_of or {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
        stage2_llm = llm or selectors.default_config(o.stage2)
        select = selectors.LLM_RERANKERS[o.stage2](text_of, stage2_llm)
    elif select is None:
        stage2_llm = None  # stage 2 off

    # 4. question index (T2-39): the tables' hypothetical questions, searched as their own lists
    by_key = {r.info.key: r for r in records}
    if not spec.hq_build:
        hq, hq_bm25 = None, None
    else:
        if hq is None or (o.hq_bm25 and hq_bm25 is None):
            text_of = text_of or {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
        if hq is None:
            hq = HqRetriever.connect(spec.hq_build, dense, records=by_key, text_of=text_of, k=o.pool,
                                     chroma=spec.chroma)
        if not o.hq_bm25:
            hq_bm25 = None
        elif hq_bm25 is None:
            rows = (files.read_questions(spec.hq_build, spec.hq_root) if spec.hq_root
                    else files.read_questions(spec.hq_build))
            hq_bm25 = BM25TableRetriever.from_index(question_index.hq_bm25(
                rows, by_key, text_of, k=len(records) if o.cond_filter else o.pool))

    # 5. 시군구 -> 시도 names for parse (P2-D9: read here once, not at import)
    if sigungu is None:
        sigungu = files.load_sigungu(spec.sigungu) if spec.sigungu else files.load_sigungu()
        if not sigungu and not _SIGUNGU_WARNED:
            _SIGUNGU_WARNED.append(True)  # once per process, as the old import-time read did
            log.warning("data/sigungu.json not found: 시군구 names in questions are not read as 시도 "
                        "(build it with `python -m kosis_rag.shared.vocab --data-dir <provided data>`)")
    return SearchResources(options=o, spec=spec, records=by_key, dense=dense, bm25=bm25,
                           text_format=fmt, stage1_score=stage1_score, select=select,
                           stage2_name=o.stage2 if o.stage2 not in OFF else "custom", stage2_llm=stage2_llm,
                           hq=hq, hq_bm25=hq_bm25, sigungu=sigungu)
