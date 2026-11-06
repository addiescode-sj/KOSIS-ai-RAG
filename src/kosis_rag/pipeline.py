"""Staged table search assembled from settings (T2-06, T2-07): one function gives the retriever the tools, the chain
and the eval runners use.

    question -> [T2-14 conditions] -> fine-tuned bge-m3 (Chroma) top 50 + BM25 top 50 -> RRF (HybridRetriever)
             -> stage 1: fine-tuned bge-reranker-v2-m3 over the top 50 (CrossEncoderReranker, when RERANK_CKPT is set)
             -> stage 2: LLM re-ranker over the top 20 (LLMReranker, STAGE2=clova | hcx | none)

Settings come from .env (SearchSettings.from_env); every stage can be switched off, so the same code measures
"none / stage 1 / stage 2 only / stage 1 + stage 2" (plans/T2-07.md). The weights are not in git (data/models/).
BM25, the re-ranker and the collection all use one table text format (kosis_rag.shared.tabletext): the collection's own
doc_format, which TEXT_FORMAT (when set) must match; weights trained on another format are refused (kosis_rag.adapters.models).

    CHROMA_COLLECTION=kosis_a_bgeft_<version>   # built by `python -m kosis_rag.index --model bge-m3-ft`
    EMBED_CKPT=data/models/bge-m3-seen-e2.pt
    RERANK_CKPT=data/models/bge-reranker-unseen-e2.pt   # empty = no stage 1
    RERANK_MODE=blend                                   # blend | score
    STAGE2=clova                                        # clova | hcx | gpt | none (HCX_THINKING=none | low for hcx)
    COND_FILTER=on                                      # on | off: T2-14 conditions before the hybrid
    TEXT_FORMAT=n2-L60                                  # optional check: n2-L60 (n3-L60 deprecated)
    HQ_BUILD=hq-<prompt8>-<tables8>-<model>-f30-k2      # T2-39 question index (kosis_rag.hq); empty = off
    HQ_WEIGHT=1.0                                       # RRF weight of the question lists
    HQ_BM25=off                                         # on: also BM25 over each table's questions

The stage 2 LLM's key (HCX_API_KEY for clova / hcx, OPENAI_API_KEY for gpt) and HCX_THINKING are read here too
(env_stage2_llm -> LLMConfig, T2-41 S3; S4 moves them to backend/app/config.py): the selectors take an LLMConfig.

The question index (T2-39) is read by the graph only (graph.build_graph adds retrieve_hq / retrieve_hq_bm25);
build_search, the nested-retriever path, ignores it so its behaviour stays as it was.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

from kosis_rag.adapters import files
from kosis_rag.legacy.rerankers import CrossEncoderReranker, LLMReranker
from kosis_rag.legacy.retrievers import BM25TableRetriever, ChromaTableRetriever, HqRetriever, HybridRetriever
from kosis_rag.llm.config import LLMConfig
from kosis_rag.parse.node import parse_query
from kosis_rag.rerank import selectors
from kosis_rag.rerank.stage1 import passage_scorer
from kosis_rag.rerank.stage2 import Score, Select
from kosis_rag.retrieve import bm25 as table_bm25  # module access: pipeline re-exports none of its names
from kosis_rag.retrieve import hq as question_index
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared import tabletext as text

REPO = Path(__file__).resolve().parents[2]
OFF = ("", "0", "off", "false", "no", "none")
# The environment variable holding each stage 2 LLM's key (T2-41 S3: read by env_stage2_llm until S4)
STAGE2_KEY_ENV = {"clova": "HCX_API_KEY", "hcx": "HCX_API_KEY", "gpt": "OPENAI_API_KEY"}
log = logging.getLogger(__name__)


class SearchSettings(BaseModel):
    tables: Path = REPO / "data" / "tables.jsonl"
    collection: str | None = None  # None -> CHROMA_COLLECTION
    text_format: str | None = None  # None -> the collection's doc_format
    embed_ckpt: str | None = None  # None -> EMBED_CKPT (checked against the collection)
    cond_filter: bool = True
    pool: int = 50  # tables each list keeps before RRF, and the hybrid hands on
    rerank_ckpt: str | None = None  # None -> no stage 1
    rerank_mode: str = "blend"
    rerank_n: int = 50
    stage2: str = "none"
    stage2_k: int = 20
    answer_k: int = 5  # candidate tables handed to gate / guide by the answer API when a request gives no k (T2-39)
    device: str | None = None
    hq_build: str | None = None  # T2-39: question index build id (HQ_BUILD); None = off
    hq_weight: float = 1.0  # RRF weight of the question lists (HQ_WEIGHT)
    hq_bm25: bool = False  # also BM25 over each table's questions (HQ_BM25)

    @classmethod
    def from_env(cls, **overrides) -> SearchSettings:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
        env = os.environ.get
        values = {
            "collection": env("CHROMA_COLLECTION") or None,
            "text_format": env("TEXT_FORMAT") or None,
            "embed_ckpt": env("EMBED_CKPT") or None,
            "cond_filter": (env("COND_FILTER") or "on").strip().lower() not in OFF,
            "rerank_ckpt": env("RERANK_CKPT") or None,
            "rerank_mode": env("RERANK_MODE") or "blend",
            "stage2": (env("STAGE2") or "none").strip().lower(),
            "answer_k": int(env("ANSWER_K") or 5),
            "hq_build": (env("HQ_BUILD") or "").strip() or None,
            "hq_weight": float(env("HQ_WEIGHT") or 1.0),
            "hq_bm25": (env("HQ_BM25") or "off").strip().lower() not in OFF,
        }
        return cls(**{**values, **overrides})

    def label(self) -> str:
        hq = ""
        if self.hq_build:
            hq = "+hq" + ("+hqbm25" if self.hq_bm25 else "") + (f"(w{self.hq_weight:g})" if self.hq_weight != 1.0 else "")
        parts = ["hybrid" + ("+cond" if self.cond_filter else "") + hq]
        if self.rerank_ckpt:
            parts.append(f"stage1-{self.rerank_mode}")
        if self.stage2 not in OFF:
            parts.append(self.stage2)
        return " -> ".join(parts)


def text_format(s: SearchSettings, dense: BaseRetriever) -> str:
    """The table text the whole search uses: the dense collection's format; TEXT_FORMAT, when set, must agree.

    In: s = search settings (text_format = TEXT_FORMAT or None); dense = the dense retriever (its doc_format, if any).
    Out: the format name used for BM25, the re-ranker passages and stage 2 passages.
    Side effects: raises ValueError when TEXT_FORMAT disagrees with the collection.
    Node: none directly (load_resources calls it once).
    """
    built = getattr(dense, "doc_format", None)
    fmt = built if built in text.FORMATS else (s.text_format or text.DOC_FORMAT)  # KURE collections: our default
    if s.text_format and s.text_format != fmt:
        raise ValueError(f"TEXT_FORMAT={s.text_format} but the collection holds {fmt!r} text")
    return fmt


@dataclass
class SearchResources:
    """Everything build_search loads once: records, dense retriever, BM25, text format, stage 1 scorer, stage 2 selector.

    Shared by the old path (build_search wraps them as nested retrievers) and the graph (graph.build_graph closes its
    nodes over them), so both search with the same objects.
    """

    settings: SearchSettings
    records: dict[str, TableRecord]  # key -> record
    dense: BaseRetriever
    bm25: BM25TableRetriever
    text_format: str
    stage1_score: Score | None  # None = stage 1 off
    select: Select | None  # None = stage 2 off
    stage2_name: str
    hq: BaseRetriever | None = None  # T2-39 question index, dense (kosis_rag.legacy.retrievers.HqRetriever); None = off
    hq_bm25: BaseRetriever | None = None  # T2-39 question index, BM25 over each table's questions; None = off
    sigungu: dict[str, list[str]] = field(default_factory=dict)  # 시군구 -> 시도 names parse reads (P2-D9, T2-41 S2)
    stage2_llm: LLMConfig | None = None  # the config `select` was built from (T2-41 S3); None = a given selector


def env_stage2_llm(name: str) -> LLMConfig:
    """The stage 2 LLM's config from this repo's .env and the environment (T2-41 S3; S4: backend/app/config.py).

    In: name = a selectors.LLM_RERANKERS key (clova | hcx | gpt). Out: selectors.default_config(name) with api_key =
        STAGE2_KEY_ENV[name] (None when unset: rerank2 raises LLMKeyMissing) and thinking = HCX_THINKING (default
        none) — the variables the selectors used to read themselves (Phase 1 load_env + get_secret, os.getenv).
    Side effects: loads this repo's .env into the environment (existing variables kept). Node: none.
    """
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env", override=False)
    return selectors.default_config(name, api_key=os.environ.get(STAGE2_KEY_ENV[name]) or None,
                                    thinking=(os.getenv("HCX_THINKING") or "none").strip().lower())


_SIGUNGU_WARNED: list[bool] = []  # set by the first load_resources that finds no sigungu.json

def load_resources(settings: SearchSettings | None = None, *, records: list[TableRecord] | None = None,
                   dense: BaseRetriever | None = None, stage1_score: Score | None = None,
                   select: Select | None = None, hq: BaseRetriever | None = None,
                   hq_bm25: BaseRetriever | None = None,
                   sigungu: dict[str, list[str]] | None = None,
                   stage2_llm: LLMConfig | None = None) -> SearchResources:
    """Load (or take the given fakes for) every heavy part of the staged search once.

    In: settings = search settings (default: .env); records, dense, stage1_score, select, hq, hq_bm25, sigungu = parts
        that replace the loaded ones (tests pass fakes here so nothing real is touched).
    Out: SearchResources. stage1_score is set when RERANK_CKPT is set or a scorer is given; select when STAGE2 names
        a known re-ranker or a selector is given. stage2_name = STAGE2, or "custom" for a given selector with STAGE2
        off. BM25 returns every table when the condition filter is on (it is filtered after the search), else `pool`.
        hq / hq_bm25 (T2-39) are set only when HQ_BUILD is set (hq_bm25 also needs HQ_BM25 on); otherwise None, even
        when given. The question BM25 follows the BM25 `k` rule above. sigungu = the 시군구 -> 시도 dictionary parse
        reads (data/sigungu.json, {} when the file is missing; T2-41 S2: it was query.SIGUNGU, read at import).
        stage2_llm = the stage 2 LLM's config when the selector is built here (T2-41 S3): the given one, else
        env_stage2_llm(STAGE2); with a given selector, the given stage2_llm (default None: rerank2 checks no key);
        None when stage 2 is off.
    Side effects: reads tables.jsonl and data/sigungu.json (one warning when it is missing), connects to Chroma and
        loads the query embedder, loads the cross-encoder, opens the question collection (reusing the dense
        retriever's query embedder) and reads its questions.jsonl, unless the matching part is given. Raises ValueError for a TEXT_FORMAT that disagrees with the collection or an
        unknown STAGE2, as build_search always did.
    Node: none (called once before build_search / build_graph).
    """
    s = settings or SearchSettings.from_env()
    # 1. registry, dense collection and the text format they agree on
    records = records if records is not None else files.read_records(s.tables)
    if dense is None:
        dense = ChromaTableRetriever.connect(s.collection, k=s.pool, device=s.device, embed_ckpt=s.embed_ckpt)
    fmt = text_format(s, dense)
    bm25 = BM25TableRetriever.from_index(  # filtered after: score all
        table_bm25.bm25_text(records, k=len(records) if s.cond_filter else s.pool, fmt=fmt))

    # 2. stage 1 scorer (fine-tuned cross-encoder)
    text_of: dict[str, str] | None = None
    if s.rerank_ckpt and stage1_score is None:
        from kosis_rag.adapters.models import load_cross_encoder

        text_of = {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
        scorer = load_cross_encoder(s.rerank_ckpt, device=s.device, doc_format=fmt)
        stage1_score = passage_scorer(scorer, text_of, text.norm_query)

    # 3. stage 2 selector (LLM re-ranker)
    if select is None and s.stage2 not in OFF:
        if s.stage2 not in selectors.LLM_RERANKERS:
            raise ValueError(f"unknown STAGE2 {s.stage2!r}; known: none, {', '.join(selectors.LLM_RERANKERS)}")
        text_of = text_of or {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
        stage2_llm = stage2_llm or env_stage2_llm(s.stage2)
        select = selectors.LLM_RERANKERS[s.stage2](text_of, stage2_llm)
    elif select is None:
        stage2_llm = None  # stage 2 off

    # 4. question index (T2-39): the tables' hypothetical questions, searched as their own lists
    by_key = {r.info.key: r for r in records}
    if not s.hq_build:
        hq, hq_bm25 = None, None
    else:
        if hq is None or (s.hq_bm25 and hq_bm25 is None):
            text_of = text_of or {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
        if hq is None:
            hq = HqRetriever.connect(s.hq_build, dense, records=by_key, text_of=text_of, k=s.pool)
        if not s.hq_bm25:
            hq_bm25 = None
        elif hq_bm25 is None:
            hq_bm25 = BM25TableRetriever.from_index(question_index.hq_bm25(
                files.read_questions(s.hq_build), by_key, text_of, k=len(records) if s.cond_filter else s.pool))

    # 5. 시군구 -> 시도 names for parse (P2-D9: read here once, not at import)
    if sigungu is None:
        sigungu = files.load_sigungu()
        if not sigungu and not _SIGUNGU_WARNED:
            _SIGUNGU_WARNED.append(True)  # once per process, as the old import-time read did
            log.warning("data/sigungu.json not found: 시군구 names in questions are not read as 시도 "
                        "(build it with `python -m kosis_rag.shared.vocab --data-dir <provided data>`)")
    return SearchResources(settings=s, records=by_key, dense=dense, bm25=bm25,
                           text_format=fmt, stage1_score=stage1_score, select=select,
                           stage2_name=s.stage2 if s.stage2 not in OFF else "custom", hq=hq, hq_bm25=hq_bm25,
                           sigungu=sigungu, stage2_llm=stage2_llm)


def build_search(settings: SearchSettings | None = None, *, records: list[TableRecord] | None = None,
                 dense: BaseRetriever | None = None, stage1_score: Score | None = None,
                 select: Select | None = None, sigungu: dict[str, str] | None = None,
                 stage2_llm: LLMConfig | None = None) -> BaseRetriever:
    """The staged retriever for `settings` (default: .env). The keyword parts replace the loaded ones (tests).

    In: as load_resources. Out: HybridRetriever (with the T2-14 condition when COND_FILTER is on), wrapped by
    CrossEncoderReranker when stage 1 is on and by LLMReranker when stage 2 is on.
    The question index (HQ_BUILD, T2-39) is graph-only: it is switched off here, so nothing of it is loaded and this
    path searches as it did before T2-39.
    Side effects: those of load_resources. Node: not used by the graph (build_graph uses load_resources directly).
    """
    settings = (settings or SearchSettings.from_env()).model_copy(update={"hq_build": None, "hq_bm25": False})
    r = load_resources(settings, records=records, dense=dense, stage1_score=stage1_score, select=select,
                       sigungu=sigungu, stage2_llm=stage2_llm)
    s = r.settings
    retriever: BaseRetriever = HybridRetriever(
        retrievers=[r.dense, r.bm25], names=["dense", "bm25"], k=s.pool, k_each=s.pool, score_from=0,
        records=r.records, condition=(lambda q: parse_query(q, sigungu=r.sigungu).to_filter()) if s.cond_filter else None)
    if r.stage1_score is not None:
        retriever = CrossEncoderReranker(base=retriever, score=r.stage1_score, n=s.rerank_n, mode=s.rerank_mode)
    if r.select is not None:
        retriever = LLMReranker(base=retriever, select=r.select, k_in=s.stage2_k, name=r.stage2_name)
    return retriever
