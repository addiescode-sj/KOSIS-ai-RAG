"""Staged search for the e2e pipeline (plan 43 port): hybrid search -> stage 1 cross-encoder -> stage 2 LLM re-ranker.

Each stage is a LangChain BaseRetriever wrapping the one before, so evals / chain.answer / the e2e runner take the
last one like any retriever (Document.metadata["key"] = table key):

    hybrid (fine-tuned bge-m3 dense + BM25, RRF)  top N (50)
      -> Stage1Retriever: fine-tuned bge-reranker-v2-m3 scores (n2 question, table text) for the top N
         mode "blend" = RRF of the re-ranker order and the input order (plan 43 dev choice, e2 S1b)
         mode "score" = re-ranker score only
      -> Stage2Retriever: an LLM re-ranker sees the top K (20) and returns the tables it cites;
         cited tables go first (stage-1 order), then the rest. A failed call keeps the stage-1 order.

Stage 2 is pluggable: SELECTORS maps a name to a factory (text_of, opts) -> select(question, keys) -> cited keys.
"clova" = CLOVA Studio /v1/api-tools/reranker with the question wrapped as in plan 42
("다음 질문의 수치를 찾을 수 있는 통계표: {q}"), every answer cached under data/lab/two_stage/clova/ (shared with
two_stage_rerank_eval.py). A hand-made LLM re-ranker is added as another SELECTORS entry.
The re-ranker text is the lab n2 table text (the text both fine-tunings were trained on), looked up by key.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

RRF_C = 60
WRAP = "다음 질문의 수치를 찾을 수 있는 통계표: {q}"


def table_texts(max_labels: int = 60) -> dict[str, str]:
    """key -> lab n2 table text (same as data/lab/rrft/passages.json)."""
    from lab_common import docs, load_records

    return {d["key"]: d["embed"] for d in docs(load_records(), "norm", max_labels)}


def resolve_reranker(path: Path) -> Path:
    """A .pt file, or a data/lab/rrft/<run> folder (its last epoch)."""
    path = Path(path)
    if path.is_dir():
        pts = sorted(path.glob("unseen-e*.pt"), key=lambda x: int(x.stem.split("-e")[-1]))
        if not pts:
            raise FileNotFoundError(f"no unseen-e*.pt in {path}")
        path = pts[-1]
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def cross_encoder_scorer(ckpt: Path | None, text_of: dict[str, str], max_length: int = 512,
                         batch: int = 16) -> Callable[[str, list[str]], list[float]]:
    """score(question, keys): base bge-reranker-v2-m3 with the fine-tuned layers of `ckpt` (None = untrained)."""
    from lab_common import query_text
    from rerank_ft_eval import load_cross_encoder

    model = load_cross_encoder(ckpt, max_length)
    cache: dict[tuple[str, str], float] = {}

    def score(question: str, keys: list[str]) -> list[float]:
        q = query_text(question, "norm")  # the n2 question, as in training and evaluation
        need = [k for k in keys if (q, k) not in cache]
        if need:
            out = model.predict([(q, text_of[k]) for k in need], batch_size=batch, show_progress_bar=False,
                                convert_to_numpy=True)
            for k, v in zip(need, out.reshape(-1).tolist()):
                cache[(q, k)] = float(v)
        return [cache[(q, k)] for k in keys]

    return score


def clova_selector(text_of: dict[str, str], opts: dict) -> Callable[[str, list[str]], list[str]]:
    from two_stage_rerank_eval import stage2_clova

    args = SimpleNamespace(api_max_tokens=opts.get("max_tokens", 512), interval=opts.get("interval", 1.0))
    state: dict = {}
    wrap = opts.get("wrap", True)

    def select(question: str, keys: list[str]) -> list[str]:
        q = WRAP.format(q=question) if wrap else question
        return stage2_clova(q, keys, text_of, args, state)["cited"]

    return select


SELECTORS: dict[str, Callable[[dict[str, str], dict], Callable[[str, list[str]], list[str]]]] = {
    "clova": clova_selector,
}


class Stage1Retriever(BaseRetriever):
    """Re-orders the base top N by a cross-encoder; the rest of the base list follows unchanged."""

    base: BaseRetriever
    score: Any  # (question, keys) -> scores
    n: int = 50
    mode: str = "blend"  # blend | score
    c: int = RRF_C

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        docs = self.base.invoke(query)
        head, tail = docs[: self.n], docs[self.n:]
        if not head:
            return docs
        t0 = time.perf_counter()
        sc = self.score(query, [d.metadata["key"] for d in head])
        by_score = sorted(range(len(head)), key=lambda i: -sc[i])
        r_rerank = {i: r for r, i in enumerate(by_score, 1)}
        if self.mode == "score":
            order = by_score
        else:  # stable: ties keep the input order
            order = sorted(range(len(head)), key=lambda i: -(1 / (self.c + r_rerank[i]) + 1 / (self.c + i + 1)))
        sec = time.perf_counter() - t0
        out = [Document(page_content=head[i].page_content, id=head[i].id,
                        metadata={**head[i].metadata, "input_rank": i + 1, "stage1_score": sc[i],
                                  "stage1_rank": r_rerank[i], "stage1_seconds": sec}) for i in order]
        return out + tail


class Stage2Retriever(BaseRetriever):
    """An LLM re-ranker picks among the base top K; its cited tables go first, in the base order."""

    base: BaseRetriever
    select: Any  # (question, keys) -> cited keys
    k_in: int = 20
    stage_name: str = "llm"

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        docs = self.base.invoke(query)
        keys = [d.metadata["key"] for d in docs[: self.k_in]]
        t0 = time.perf_counter()
        try:
            cited, error = set(self.select(query, keys)) & set(keys), None
        except Exception as exc:  # the answer still gets the stage-1 order
            cited, error = set(), f"{type(exc).__name__}: {exc}"
        meta = {"stage2": self.stage_name, "stage2_cited_n": len(cited), "stage2_error": error,
                "stage2_seconds": time.perf_counter() - t0}
        first = [d for d in docs if d.metadata["key"] in cited]
        rest = [d for d in docs if d.metadata["key"] not in cited]
        return [Document(page_content=d.page_content, id=d.id,
                         metadata={**d.metadata, **meta, "stage2_cited": d.metadata["key"] in cited})
                for d in first + rest]


def build_staged(base: BaseRetriever, reranker: Path | None, stage2: str | None, *, mode: str = "blend",
                 n: int = 50, k_in: int = 20, max_length: int = 512, interval: float = 1.0,
                 wrap: bool = True) -> tuple[BaseRetriever, str]:
    """base (top >= n) -> stage 1 (when reranker) -> stage 2 (when stage2). Returns the retriever and a label."""
    text_of = table_texts()
    retriever, label = base, []
    if reranker is not None:
        ckpt = resolve_reranker(reranker)
        retriever = Stage1Retriever(base=retriever, score=cross_encoder_scorer(ckpt, text_of, max_length), n=n, mode=mode)
        label.append(f"bge-reranker ft {ckpt.parent.name}/{ckpt.stem} ({mode}, top {n})")
    if stage2 and stage2 != "none":
        if stage2 not in SELECTORS:
            raise ValueError(f"unknown stage 2 {stage2!r}; known: none, {', '.join(SELECTORS)}")
        select = SELECTORS[stage2](text_of, {"interval": interval, "wrap": wrap})
        retriever = Stage2Retriever(base=retriever, select=select, k_in=k_in, stage_name=stage2)
        label.append(f"{stage2} re-ranker top {k_in}{' wrapped' if wrap else ''}")
    return retriever, " -> ".join(label)
