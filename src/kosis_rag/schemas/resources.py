"""Settings objects the search module is built from (T2-41 S1: ChromaConfig; S4: SearchOptions, IndexSpec and
SearchResources from pipeline.py — plans/T2-41.md 입출력 계약).

The caller declares them (§0.6): nothing here reads .env or the environment. SearchOptions is how the search behaves,
IndexSpec is where its data and models are; their defaults are the old settings object's defaults. SearchResources is
what adapters.loader.load_resources opens from them once (records, dense, BM25, scorers, question index).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # annotation only: schemas does not import llm/ at run time
    from kosis_rag.llm.config import LLMConfig

OFF = ("", "0", "off", "false", "no", "none")  # values that switch a stage off (STAGE2, COND_FILTER, HQ_BM25)


class ChromaConfig(BaseModel):
    """Where the shared ChromaDB server is. Defaults match `docker compose up -d chroma` on the host (port 8001);
    inside compose the backend uses host "chroma", port 8000. The caller builds it (no .env read here)."""

    host: str = "localhost"
    port: int = 8001
    collection: str | None = None  # None: the caller names the collection


class IndexSpec(BaseModel):
    """Paths, connection and weights of one search setup (T2-41 S4). Fake-resource tests leave it empty.

    A None path means the repo default the old settings object used (data/tables.jsonl, data/sigungu.json, data/hq);
    a None checkpoint means that stage is off (rerank_ckpt) or the collection's own (embed_ckpt is checked against it).
    """

    tables: Path | None = None  # data/tables.jsonl (None: adapters.files.TABLES)
    sigungu: Path | None = None  # data/sigungu.json (None: adapters.files default; P2-D9: parse reads resources.sigungu)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)  # server and dense collection (CHROMA_COLLECTION)
    text_format: str | None = None  # None = the collection's doc_format (TEXT_FORMAT must agree when set)
    embed_ckpt: str | None = None  # fine-tuned bge-m3 query embedder (EMBED_CKPT)
    rerank_ckpt: str | None = None  # None = no stage 1 (RERANK_CKPT)
    hq_build: str | None = None  # T2-39 question index build id (HQ_BUILD); None = off
    hq_root: Path | None = None  # data/hq (None: adapters.files.HQ_DIR)
    device: str | None = None  # cpu | cuda; None = auto


class SearchOptions(BaseModel):
    """How the search behaves (T2-41 S4). Defaults = the old settings object's defaults."""

    cond_filter: bool = True  # T2-14 conditions before the lists (COND_FILTER)
    pool: int = 50  # tables each list keeps before RRF, and the fusion hands on
    rerank_mode: str = "blend"  # stage 1: blend | score (RERANK_MODE)
    rerank_n: int = 50
    stage2: str = "none"  # clova | hcx | gpt | none (STAGE2)
    stage2_k: int = 20
    answer_k: int = 5  # candidate tables handed to gate / guide when a request gives no k (ANSWER_K, T2-39)
    hq_weight: float = 1.0  # RRF weight of the question lists (HQ_WEIGHT)
    hq_bm25: bool = False  # also BM25 over each table's questions (HQ_BM25)

    def label(self, spec: IndexSpec) -> str:
        """The setup's short name, as the old settings' label() gave it: "hybrid+cond+hq -> stage1-blend -> hcx"."""
        hq = ""
        if spec.hq_build:
            hq = "+hq" + ("+hqbm25" if self.hq_bm25 else "") + (f"(w{self.hq_weight:g})" if self.hq_weight != 1.0 else "")
        parts = ["hybrid" + ("+cond" if self.cond_filter else "") + hq]
        if spec.rerank_ckpt:
            parts.append(f"stage1-{self.rerank_mode}")
        if self.stage2 not in OFF:
            parts.append(self.stage2)
        return " -> ".join(parts)


@dataclass
class SearchResources:
    """Everything load_resources opens once: records, dense list, BM25, text format, stage 1 scorer, stage 2 selector,
    question index, 시군구 dictionary — with the options and spec they were opened for (T2-41 S4: was
    pipeline.SearchResources with `settings`).

    Shared by the old path (legacy.pipeline.build_search wraps them as nested retrievers) and the graph (graph.search
    closes its nodes over them), so both search with the same objects. The step functions read `options`.
    """

    options: SearchOptions
    spec: IndexSpec
    records: dict[str, Any]  # key -> TableRecord
    dense: Any  # ChromaTableRetriever or a stand-in (with_filter, embed_query)
    bm25: Any  # BM25TableRetriever
    text_format: str
    stage1_score: Any | None  # rerank.stage2.Score; None = stage 1 off
    select: Any | None  # rerank.stage2.Select; None = stage 2 off
    stage2_name: str
    stage2_llm: LLMConfig | None = None  # the config `select` was built from (T2-41 S3); None = a given selector
    hq: Any | None = None  # T2-39 question index, dense (legacy.retrievers.HqRetriever); None = off
    hq_bm25: Any | None = None  # T2-39 question index, BM25 over each table's questions; None = off
    sigungu: dict[str, list[str]] = field(default_factory=dict)  # 시군구 -> 시도 names parse reads (P2-D9)
