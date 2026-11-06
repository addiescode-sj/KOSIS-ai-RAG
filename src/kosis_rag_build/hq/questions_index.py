"""Load a question build into Chroma (T2-39): one Chroma document per hypothetical question (T2-41 S6a: the CLI of the
old hq/index.py as nodes; the graph is graph/index_build.build_questions_graph).

    python -m kosis_rag_build.cli hq-index --build <build_id> --ckpt data/models/unseen-e1.pt
    python -m kosis_rag_build.cli hq-index --build <build_id> --ckpt data/models/unseen-e1.pt --desc   # S1-desc ablation
    python -m kosis_rag_build.cli hq-index --build <build_id> --ckpt ... --dry-run   # data_sha256, name, count only

Questions are embedded with the same fine-tuned embedder as the service's table collection after the same surface
rules (tabletext.norm_query). Collection `kosis_hq__<build_id>__<ckpt8>`; its metadata records embed_model, embed_ckpt,
doc_format "hq" and hq_build, and the search (kosis_rag.retrieve.hq, legacy HqRetriever.connect) refuses a collection
embedded with other weights than the dense side. `--desc` indexes each table's one-sentence description instead
(build id `<build_id>-desc`).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from kosis_rag.adapters import files
from kosis_rag.adapters import models as model
from kosis_rag.shared.hqnames import HQ_DOC_FORMAT, collection_name  # the name the search opens
from kosis_rag.shared import tabletext as text
from kosis_rag_build.chroma import load
from kosis_rag_build.schemas import QuestionsIndexState


def data_sha256(rows: list[dict]) -> str:
    """sha256 of the rows to index ("<id>\\t<text>" per row, newline-joined): the collection's data_sha256.

    Side effects: none. Node: read_questions.
    """
    return hashlib.sha256("\n".join(f"{r['id']}\t{r['text']}" for r in rows).encode("utf-8")).hexdigest()


def _timed(name: str, trace: dict, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
    return {**trace, name: {"seconds": time.perf_counter() - started, **extra}}


def read_questions(state: QuestionsIndexState) -> dict:
    """The build's rows, their texts after norm_query and data_sha256; on a dry run also the weights id and the
    collection name (models.ckpt_id of the checkpoint file: no model is loaded).

    In: state.build, state.hq_root, state.ckpt, state.collection, state.dry_run.
    Out: {"rows", "texts", "data_sha256", ["ckpt_id", "name"], "trace"}.
    Side effects: reads the build folder; SystemExit when the build has no rows. Node: read_questions.
    """
    started = time.perf_counter()
    rows = files.read_questions(state.build, *([state.hq_root] if state.hq_root else []))
    if not rows:
        raise SystemExit(f"build {state.build} has no rows")
    out: dict[str, Any] = {"rows": rows, "texts": [text.norm_query(r["text"]) for r in rows],
                           "data_sha256": data_sha256(rows)}
    if state.dry_run:
        ckpt = model.ckpt_id(model.resolve_ckpt(state.ckpt))
        out |= {"ckpt_id": ckpt, "name": state.collection or collection_name(state.build, ckpt)}
    return {**out, "trace": _timed("read_questions", state.trace, started, rows=len(rows))}


def make_embed(embedder: Any = None):
    """The embed node over `embedder` (an object with embed_documents, name, ckpt — models.Embedder or a fake; None:
    models.load_embedder(state.ckpt, ...) when the node runs)."""

    def embed(state: QuestionsIndexState) -> dict:
        """In: state.texts, rows, build. Out: {"vectors", "ckpt_id", "name", "record", "trace"}.
        Side effects: loads and runs the embedder. Node: embed."""
        started = time.perf_counter()
        e = embedder or model.load_embedder(state.ckpt, device=state.device, doc_format=text.DOC_FORMAT)
        vectors = e.embed_documents(state.texts)
        name = state.collection or collection_name(state.build, e.ckpt)
        rows = state.rows
        record = {"index_version": state.index_version, "doc_format": HQ_DOC_FORMAT, "hq_build": state.build,
                  "embed_model": e.name, "embed_ckpt": e.ckpt, "data_sha256": state.data_sha256,
                  "dim": len(vectors[0]), "normalized": True, "hnsw:space": "cosine", "n_docs": len(rows),
                  "n_tables": len({r["key"] for r in rows})}
        return {"vectors": vectors, "ckpt_id": e.ckpt, "name": name, "record": record,
                "trace": _timed("embed", state.trace, started)}

    return embed


def make_load(client: Any = None):
    """The load node: kosis_rag_build.chroma.load through `client` (None: kosis_rag.adapters.chroma.chroma_client(
    state.chroma), opened when the node runs)."""

    def load_node(state: QuestionsIndexState) -> dict:
        """In: state.rows, texts, vectors, name, record. Out: {"count", "trace"}. Side effects: writes the
        collection. Node: load."""
        started = time.perf_counter()
        c = client
        if c is None:
            from kosis_rag.adapters.chroma import chroma_client

            c = chroma_client(state.chroma)
        rows = state.rows
        count = load(c, [r["id"] for r in rows], state.texts, [{"key": r["key"], "type": r["type"]} for r in rows],
                     state.vectors, state.name, state.record)
        return {"count": count, "trace": _timed("load", state.trace, started)}

    return load_node


def summary(state: QuestionsIndexState, log: Callable[[str], None] = print) -> None:
    """The CLI lines after a run (dry run: what would be loaded)."""
    n_tables = len({r["key"] for r in state.rows})
    if state.dry_run:
        log(f"dry run (no embedding, no Chroma): collection {state.name}, {len(state.rows)} questions over {n_tables} "
            f"tables, data_sha256 {state.data_sha256}, embedder {state.ckpt_id}")
        return
    log(f"collection {state.name}: {state.count} questions over {state.record['n_tables']} tables "
        f"(embedder {state.ckpt_id})")
    log(f"to search with it, set HQ_BUILD={state.build} (empty = off); HQ_BM25=on adds the question BM25")
