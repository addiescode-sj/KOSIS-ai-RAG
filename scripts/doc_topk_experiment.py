"""T2-21 (KOS-83): table-document fields, chunks of the long text and Top-K, compared on the T2-02 golden set.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/doc_topk_experiment.py topk     # Chroma build + BM25 + RRF, k curve
    ... python scripts/doc_topk_experiment.py fields   # one field group dropped at a time (in-memory KURE-v1)
    ... python scripts/doc_topk_experiment.py regions  # region names put back (시도, 시군구, region labels), dev and test
    ... python scripts/doc_topk_experiment.py chunk    # search_text chunks of the long tables, max-pool and RRF
    ... python scripts/doc_topk_experiment.py confirm --drop <field|none> --chunk <name|none>   # adopted set on test

Variants are compared with each other on in-memory cosine ranking only; the Chroma build (HNSW) is measured once
for the k curve and never mixed with the in-memory numbers. Scores follow evals.evaluators.retrieval: a question
counts as a hit at k when any gold key of any slot is in the top k. Results go to data/doc_topk/ (gitignored);
the report keeps aggregates only, no question text or gold keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import cache
from pathlib import Path

import numpy as np

from evals.evaluators.retrieval import best_tau, first_rank, predict_no_answer, prf
from evals.schemas import RetrievalItem, load_items
from kosis_rag.index import (EMBED_MODEL, LABEL_CAP, REPO, clean_item, read_records, region_summary, sido_names,
                             to_document)
from kosis_rag.roles import table_roles
from kosis_rag.schemas import TableClass, TableRecord
from kosis_rag.vocab import load_sigungu

OUT = REPO / "data" / "doc_topk"
KS = (1, 2, 3, 5, 10, 15, 20, 30, 50)
DEPTH = max(KS)
RRF_C = 60  # same constant as retriever.HybridRetriever
LONG_CHARS = 2000  # search_text longer than this is cut down in the embedding document
FIELDS = ("survey", "labels", "items", "measure")  # field groups dropped one at a time
REGION_ADDS = ("sido", "sigungu", "regionlabels")  # region names put back, one variant each (T2-21 follow-up)
CHUNKS = {"c500": (500, 50), "c1000": (1000, 100)}  # chunk size, overlap
SHOW_SLOPE = 0.2  # adopt the first k after which each extra k adds fewer than 0.2 questions (1 per 5)


def sigungu_names(labels: list[str]) -> list[str]:
    """The 시군구 in a label list (names in data/sigungu.json), each once."""
    known = _sigungu()
    return list(dict.fromkeys(label for label in labels if label in known))


@cache
def _sigungu() -> frozenset[str]:
    return frozenset(load_sigungu())


def variant_doc(record: TableRecord, drop: frozenset[str] = frozenset(), add: frozenset[str] = frozenset()) -> str:
    """index.to_document with field groups left out or region names put back; with neither it is the loaded
    document, byte for byte. add: "sido" names the 시도 in the region summary (T2-25 trial), "sigungu" also lists
    the 시군구 names, "regionlabels" lists region labels like the other classes (as before T2-25)."""
    info = record.info
    cleaned = [clean_item(i) for i in record.items]
    items = list(dict.fromkeys(name for name, _ in cleaned))
    units = list(dict.fromkeys(unit for _, unit in cleaned if unit))
    sido = {c.name: sido_names(c.labels) for c in record.classes} if add & {"sido", "sigungu"} else {}
    regions = " / ".join(region_summary(c.levels, sido.get(c.name)) for c in record.classes if c.levels) or "지역 구분 없음"
    measure, _ = table_roles(record)
    counted = f" | 세는 대상: {measure}" if measure and "measure" not in drop else ""
    head_items = "" if "items" in drop else f" | 항목: {', '.join(items[:8])}"
    survey = "" if "survey" in drop else f"조사명: {info.survey_name} | "
    lines = [
        f"{info.table_name}{counted} | {'·'.join(record.prd_se)} {record.start}~{record.end} | 지역 단위: {regions}{head_items}",
        f"통계표: {info.table_name}",
        f"{survey}작성기관: {info.organization}" + (f" | 주제: {record.topic}" if record.topic else ""),
        f"수록기간: {record.start}~{record.end} ({'·'.join(record.prd_se)}), 목록 표기 {info.period}",
        "분류 경로: " + " / ".join(record.menu_paths),
    ]
    if "items" not in drop:
        lines.append("항목: " + ", ".join(items))
    if units:
        lines.append("단위: " + ", ".join(units))
    for c in record.classes:
        if c.levels:
            lines.append(f"{c.name}: 지역 단위 {region_summary(c.levels, sido.get(c.name))}")
            if "sigungu" in add and (names := sigungu_names(c.labels)):
                lines.append("시군구: " + ", ".join(names))
            if "regionlabels" in add:
                lines.append(_listed(c))
        elif "labels" in drop:
            lines.append(c.name)
        else:
            lines.append(_listed(c))
    return "\n".join(lines)


def _listed(c: TableClass) -> str:
    shown = c.labels[:LABEL_CAP]
    more = f" 외 {c.n_labels - len(shown)}개" if c.n_labels > len(shown) else ""
    return f"{c.name}: " + ", ".join(shown) + more


def chunk_texts(records: list[TableRecord], size: int, overlap: int) -> tuple[list[str], list[str]]:
    """(chunk texts, table key of each chunk) for the tables whose search_text is longer than LONG_CHARS."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    texts, keys = [], []
    for r in records:
        if len(r.search_text) > LONG_CHARS:
            parts = splitter.split_text(r.search_text)
            texts += parts
            keys += [r.info.key] * len(parts)
    return texts, keys


def rank_keys(scores: np.ndarray, keys: list[str], depth: int = DEPTH) -> list[str]:
    top = np.argsort(-scores, kind="stable")[:depth]
    return [keys[i] for i in top]


def max_pool(doc_scores: np.ndarray, keys: list[str], chunk_scores: np.ndarray, chunk_keys: list[str]) -> np.ndarray:
    """Table score = the best of its document and its chunks."""
    out = doc_scores.copy()
    pos = {k: i for i, k in enumerate(keys)}
    for s, k in zip(chunk_scores, chunk_keys):
        out[pos[k]] = max(out[pos[k]], s)
    return out


def rrf(lists: list[list[str]], c: int = RRF_C) -> list[str]:
    """Reciprocal rank fusion of ranked key lists (a key counts once per list, at its best rank)."""
    score: dict[str, float] = {}
    for keys in lists:
        for rank, key in enumerate(dict.fromkeys(keys), 1):
            score[key] = score.get(key, 0.0) + 1.0 / (c + rank)
    return sorted(score, key=lambda k: -score[k])


def curve(items: list[RetrievalItem], ranked: dict[str, list[str]]) -> dict:
    """Hit@k for every k in KS and MRR@10 over the answerable items."""
    ranks = [first_rank(i, ranked[i.qid]) for i in items]
    out = {"n": len(items), **{f"hit@{k}": sum(1 for r in ranks if r and r <= k) for k in KS}}
    out["mrr@10"] = round(sum(1 / r for r in ranks if r and r <= 10) / max(len(items), 1), 4)
    return out


def show_k(c: dict, slope: float = SHOW_SLOPE) -> int:
    """The first k after which the next step adds fewer than `slope` questions per extra k."""
    for a, b in zip(KS, KS[1:]):
        if (c[f"hit@{b}"] - c[f"hit@{a}"]) / (b - a) < slope:
            return a
    return KS[-1]


# ---------- data, model, cache ----------

def golden(records: dict[str, TableRecord]) -> list[RetrievalItem]:
    return load_items(keys=set(records))


def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed(model, texts: list[str], name: str) -> np.ndarray:
    """L2-normalized vectors, cached in data/doc_topk/<name>.npy by the hash of the texts."""
    digest = hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:12]
    path = OUT / f"{name}-{digest}.npy"
    if path.exists():
        return np.load(path)
    started = time.perf_counter()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        vecs[idx] = model.encode([texts[i] for i in idx], batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    print(f"  embedded {name}: {len(texts)} texts in {time.perf_counter() - started:.0f}s")
    return vecs


def split(items: list[RetrievalItem], name: str) -> list[RetrievalItem]:
    return [i for i in items if i.split == name and i.answerable]


def save(name: str, result: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT / f'{name}.json'}")


# ---------- commands ----------

def cmd_topk(records: list[TableRecord], items: list[RetrievalItem]) -> None:
    """k curve of the shared Chroma build (KURE-v1), BM25 over search_text and their RRF; no-answer P/R per k."""
    from langchain_core.documents import Document

    from kosis_rag.retriever import BM25TableRetriever, ChromaTableRetriever

    chroma = ChromaTableRetriever.connect(k=DEPTH)
    print(f"collection: {chroma.collection.name}")
    bm25 = BM25TableRetriever.from_documents(
        [Document(page_content=r.search_text, metadata={"key": r.info.key}, id=r.info.key) for r in records], k=DEPTH)
    ranked: dict[str, dict[str, list[str]]] = {"chroma": {}, "bm25": {}, "rrf": {}}
    top1: dict[str, float | None] = {}
    for i in items:
        docs = chroma.invoke(i.question)
        ranked["chroma"][i.qid] = [d.metadata["key"] for d in docs]
        top1[i.qid] = docs[0].metadata["score"] if docs else None
        ranked["bm25"][i.qid] = [d.metadata["key"] for d in bm25.invoke(i.question)]
        ranked["rrf"][i.qid] = rrf([ranked["chroma"][i.qid], ranked["bm25"][i.qid]])[:DEPTH]
    result = {"collection": chroma.collection.name, "curves": {}, "show_k": {}}
    for name, lists in ranked.items():
        for sp in ("dev", "test"):
            c = curve(split(items, sp), lists)
            result["curves"][f"{name}/{sp}"] = c
            print(f"{name:6s} {sp:4s} " + " ".join(f"@{k}:{c[f'hit@{k}']}" for k in KS) + f"  mrr@10 {c['mrr@10']}")
        result["show_k"][name] = show_k(result["curves"][f"{name}/dev"])
    # no answer: the rule reads the top-1 score only, so it is the same at every k; tau on dev, applied to test
    dev = [i for i in items if i.split == "dev"]
    tau = best_tau([(top1[i.qid], not i.answerable) for i in dev])
    test = [i for i in items if i.split == "test"]
    result["no_answer"] = {"tau": tau, "test": prf([(predict_no_answer(top1[i.qid], tau), not i.answerable) for i in test])}
    print(f"show k (dev): {result['show_k']}  no-answer (chroma top-1, test): {result['no_answer']}")
    save("topk", result)


def cmd_fields(records: list[TableRecord], items: list[RetrievalItem]) -> None:
    model = _model()
    keys = [r.info.key for r in records]
    dev = split(items, "dev")
    q = embed(model, [i.question for i in items], "questions")
    qpos = {i.qid: n for n, i in enumerate(items)}
    base_docs = [variant_doc(r) for r in records]
    if base_docs != [to_document(r) for r in records]:
        raise SystemExit("variant_doc() no longer matches index.to_document(); fix it before comparing")
    result = {}
    for drop in ((), *((f,) for f in FIELDS)):
        name = "base" if not drop else "drop-" + drop[0]
        docs = base_docs if not drop else [variant_doc(r, frozenset(drop)) for r in records]
        vecs = embed(model, docs, f"doc-{name}")
        sims = q @ vecs.T
        ranked = {i.qid: rank_keys(sims[qpos[i.qid]], keys) for i in dev}
        c = curve(dev, ranked) | {"doc_chars_mean": round(sum(map(len, docs)) / len(docs))}
        result[name] = c
        print(f"{name:14s} " + " ".join(f"@{k}:{c[f'hit@{k}']}" for k in (1, 5, 10, 20)) + f"  mrr@10 {c['mrr@10']}  chars {c['doc_chars_mean']}")
    save("fields", result)


def cmd_regions(records: list[TableRecord], items: list[RetrievalItem]) -> None:
    """Region names put back into the document, one variant each, on dev and test (in-memory KURE-v1)."""
    model = _model()
    keys = [r.info.key for r in records]
    q = embed(model, [i.question for i in items], "questions")
    qpos = {i.qid: n for n, i in enumerate(items)}
    result = {}
    for add in ((), *((a,) for a in REGION_ADDS)):
        name = "base" if not add else "add-" + add[0]
        docs = [variant_doc(r, add=frozenset(add)) for r in records]
        sims = q @ embed(model, docs, f"doc-{name}").T
        for sp in ("dev", "test"):
            part = split(items, sp)
            c = curve(part, {i.qid: rank_keys(sims[qpos[i.qid]], keys) for i in part})
            c["doc_chars_mean"] = round(sum(map(len, docs)) / len(docs))
            result[f"{name}/{sp}"] = c
            print(f"{name:17s} {sp:4s} " + " ".join(f"@{k}:{c[f'hit@{k}']}" for k in (1, 5, 10, 20))
                  + f"  mrr@10 {c['mrr@10']}  chars {c['doc_chars_mean']}")
    save("regions", result)


def cmd_chunk(records: list[TableRecord], items: list[RetrievalItem]) -> None:
    model = _model()
    keys = [r.info.key for r in records]
    long = {r.info.key for r in records if len(r.search_text) > LONG_CHARS}
    dev = split(items, "dev")
    q = embed(model, [i.question for i in items], "questions")
    qpos = {i.qid: n for n, i in enumerate(items)}
    doc_sims = q @ embed(model, [to_document(r) for r in records], "doc-base").T
    base = {i.qid: rank_keys(doc_sims[qpos[i.qid]], keys) for i in dev}
    result = {"long_tables": len(long), "base": curve(dev, base)}
    for name, (size, overlap) in CHUNKS.items():
        texts, ckeys = chunk_texts(records, size, overlap)
        csims = q @ embed(model, texts, f"chunk-{name}").T
        for how in ("maxpool", "rrf"):
            ranked = {}
            for i in dev:
                n = qpos[i.qid]
                if how == "maxpool":
                    ranked[i.qid] = rank_keys(max_pool(doc_sims[n], keys, csims[n], ckeys), keys)
                else:
                    chunk_list = [ckeys[j] for j in np.argsort(-csims[n], kind="stable")]
                    ranked[i.qid] = rrf([rank_keys(doc_sims[n], keys, len(keys)), chunk_list])[:DEPTH]
            c = curve(dev, ranked)
            # skew: how often the long tables push in, and how many questions moved down
            pushed = [len((set(ranked[i.qid][:10]) - set(base[i.qid][:10])) & long) for i in dev]
            worse = sum(1 for i in dev if (first_rank(i, ranked[i.qid]) or 999) > (first_rank(i, base[i.qid]) or 999))
            better = sum(1 for i in dev if (first_rank(i, ranked[i.qid]) or 999) < (first_rank(i, base[i.qid]) or 999))
            c |= {"chunks": len(texts), "long_new_in_top10_mean": round(sum(pushed) / len(dev), 2),
                  "questions_up": better, "questions_down": worse}
            result[f"{name}/{how}"] = c
            print(f"{name}/{how:7s} " + " ".join(f"@{k}:{c[f'hit@{k}']}" for k in (1, 5, 10, 20))
                  + f"  mrr@10 {c['mrr@10']}  chunks {len(texts)}  long new in top10 {c['long_new_in_top10_mean']}  up {better} down {worse}")
    save("chunk", result)


def cmd_confirm(records: list[TableRecord], items: list[RetrievalItem], drop: str, chunk: str) -> None:
    """The adopted set on test (and dev for reference), in-memory, next to the in-memory base."""
    model = _model()
    keys = [r.info.key for r in records]
    q = embed(model, [i.question for i in items], "questions")
    qpos = {i.qid: n for n, i in enumerate(items)}
    base_sims = q @ embed(model, [to_document(r) for r in records], "doc-base").T
    name = "base" if drop == "none" else f"drop-{drop}"
    docs = [variant_doc(r, frozenset() if drop == "none" else frozenset({drop})) for r in records]
    sims = q @ embed(model, docs, f"doc-{name}").T
    if chunk != "none":
        texts, ckeys = chunk_texts(records, *CHUNKS[chunk])
        csims = q @ embed(model, texts, f"chunk-{chunk}").T
    result = {"adopted": {"drop": drop, "chunk": chunk}}
    for sp in ("dev", "test"):
        part = split(items, sp)
        result[f"base/{sp}"] = curve(part, {i.qid: rank_keys(base_sims[qpos[i.qid]], keys) for i in part})
        ranked = {}
        for i in part:
            n = qpos[i.qid]
            s = sims[n] if chunk == "none" else max_pool(sims[n], keys, csims[n], ckeys)
            ranked[i.qid] = rank_keys(s, keys)
        result[f"adopted/{sp}"] = curve(part, ranked)
        for label in ("base", "adopted"):
            c = result[f"{label}/{sp}"]
            print(f"{label:8s} {sp:4s} " + " ".join(f"@{k}:{c[f'hit@{k}']}" for k in KS) + f"  mrr@10 {c['mrr@10']}")
    save("confirm", result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("topk", "fields", "regions", "chunk", "confirm"))
    parser.add_argument("--drop", default="none", choices=("none", *FIELDS))
    parser.add_argument("--chunk", default="none", choices=("none", *CHUNKS))
    args = parser.parse_args(argv)
    records = read_records(REPO / "data" / "tables.jsonl")
    items = golden({r.info.key: r for r in records})
    if args.cmd == "topk":
        cmd_topk(records, items)
    elif args.cmd == "fields":
        cmd_fields(records, items)
    elif args.cmd == "regions":
        cmd_regions(records, items)
    elif args.cmd == "chunk":
        cmd_chunk(records, items)
    else:
        cmd_confirm(records, items, args.drop, args.chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
