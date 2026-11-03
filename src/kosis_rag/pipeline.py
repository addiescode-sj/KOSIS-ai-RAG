"""Staged table search assembled from settings (T2-06, T2-07): one function gives the retriever the tools, the chain
and the eval runners use.

    question -> [T2-14 conditions] -> fine-tuned bge-m3 (Chroma) top 50 + BM25 top 50 -> RRF (HybridRetriever)
             -> stage 1: fine-tuned bge-reranker-v2-m3 over the top 50 (CrossEncoderReranker, when RERANK_CKPT is set)
             -> stage 2: LLM re-ranker over the top 20 (LLMReranker, STAGE2=clova | hcx | none)

Settings come from .env (SearchSettings.from_env); every stage can be switched off, so the same code measures
"none / stage 1 / stage 2 only / stage 1 + stage 2" (plans/T2-07.md). The weights are not in git (data/models/).
BM25, the re-ranker and the collection all use one table text format (kosis_rag.text): the collection's own
doc_format, which TEXT_FORMAT (when set) must match; weights trained on another format are refused (kosis_rag.model).

    CHROMA_COLLECTION=kosis_a_bgeft_<version>   # built by `python -m kosis_rag.index --model bge-m3-ft`
    EMBED_CKPT=data/models/bge-m3-seen-e2.pt
    RERANK_CKPT=data/models/bge-reranker-unseen-e2.pt   # empty = no stage 1
    RERANK_MODE=blend                                   # blend | score
    STAGE2=clova                                        # clova | hcx | none (HCX_THINKING=none | low for hcx)
    COND_FILTER=on                                      # on | off: T2-14 conditions before the hybrid
    TEXT_FORMAT=n3-L60                                  # optional check: n3-L60 | n2-L60
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

from kosis_rag import text
from kosis_rag.loader import to_document
from kosis_rag.rerank import LLM_RERANKERS, CrossEncoderReranker, LLMReranker, Score, Select, passage_scorer
from kosis_rag.retriever import BM25TableRetriever, ChromaTableRetriever, HybridRetriever
from kosis_rag.schemas import TableRecord

REPO = Path(__file__).resolve().parents[2]
OFF = ("", "0", "off", "false", "no", "none")


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
    device: str | None = None

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
        }
        return cls(**{**values, **overrides})

    def label(self) -> str:
        parts = ["hybrid" + ("+cond" if self.cond_filter else "")]
        if self.rerank_ckpt:
            parts.append(f"stage1-{self.rerank_mode}")
        if self.stage2 not in OFF:
            parts.append(self.stage2)
        return " -> ".join(parts)


def read_records(path: Path) -> list[TableRecord]:
    with path.open(encoding="utf-8") as f:
        return [TableRecord.model_validate_json(line) for line in f if line.strip()]


def bm25_text(records: list[TableRecord], k: int, fmt: str = text.DOC_FORMAT) -> BM25TableRetriever:
    """BM25 over the full table text (every label) of `fmt`, questions with the same surface rules (lab F0)."""
    docs = []
    for r in records:
        catalog = to_document(r.info)
        docs.append(Document(page_content=text.build_doc(r, fmt)["bm25_text"], metadata=catalog.metadata, id=catalog.id))
    return BM25TableRetriever.from_documents(docs, k=k, query_text=text.norm_query)


def text_format(s: SearchSettings, dense: BaseRetriever) -> str:
    """The table text the whole search uses: the dense collection's format; TEXT_FORMAT, when set, must agree."""
    built = getattr(dense, "doc_format", None)
    fmt = built if built in text.FORMATS else (s.text_format or text.DOC_FORMAT)  # KURE collections: our default
    if s.text_format and s.text_format != fmt:
        raise ValueError(f"TEXT_FORMAT={s.text_format} but the collection holds {fmt!r} text")
    return fmt


def build_search(settings: SearchSettings | None = None, *, records: list[TableRecord] | None = None,
                 dense: BaseRetriever | None = None, stage1_score: Score | None = None,
                 select: Select | None = None) -> BaseRetriever:
    """The staged retriever for `settings` (default: .env). The keyword parts replace the loaded ones (tests)."""
    from kosis_rag.query import parse_query

    s = settings or SearchSettings.from_env()
    records = records if records is not None else read_records(s.tables)
    by_key = {r.info.key: r for r in records}
    if dense is None:
        dense = ChromaTableRetriever.connect(s.collection, k=s.pool, device=s.device, embed_ckpt=s.embed_ckpt)
    fmt = text_format(s, dense)
    bm25 = bm25_text(records, k=len(records) if s.cond_filter else s.pool, fmt=fmt)  # filtered after: score all
    retriever: BaseRetriever = HybridRetriever(
        retrievers=[dense, bm25], names=["dense", "bm25"], k=s.pool, k_each=s.pool, score_from=0, records=by_key,
        condition=(lambda q: parse_query(q).to_filter()) if s.cond_filter else None)

    text_of: dict[str, str] | None = None
    if s.rerank_ckpt or stage1_score is not None:
        if stage1_score is None:
            from kosis_rag.model import load_cross_encoder

            text_of = {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
            scorer = load_cross_encoder(s.rerank_ckpt, device=s.device, doc_format=fmt)
            stage1_score = passage_scorer(scorer, text_of, text.norm_query)
        retriever = CrossEncoderReranker(base=retriever, score=stage1_score, n=s.rerank_n, mode=s.rerank_mode)

    if select is None and s.stage2 not in OFF:
        if s.stage2 not in LLM_RERANKERS:
            raise ValueError(f"unknown STAGE2 {s.stage2!r}; known: none, {', '.join(LLM_RERANKERS)}")
        text_of = text_of or {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
        select = LLM_RERANKERS[s.stage2](text_of, {})
    if select is not None:
        retriever = LLMReranker(base=retriever, select=select, k_in=s.stage2_k,
                                name=s.stage2 if s.stage2 not in OFF else "custom")
    return retriever
