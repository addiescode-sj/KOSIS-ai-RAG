"""Compare embedding models and text recipes without a database: exact cosine top-k with numpy.

Same questions, same metrics as chroma_eval_p1.py, but document vectors are cached as .npy files, so trying
another model, rule version or label cap only embeds what is new. At 2,364 tables an exact search is the
ground truth that Chroma's HNSW index approximates, so the ranks match a Chroma run.

    python lab/embed_eval.py                                   # bge-m3, raw vs norm (labels > 60 summarized)
    python lab/embed_eval.py --max-labels 60 100 0             # label cap variants (0 = never summarize)
    python lab/embed_eval.py --models BAAI/bge-m3 intfloat/multilingual-e5-large
    python lab/embed_eval.py --show-changed

Cache: data/lab/emb/{model}__{text}__{sha8}-T{max_tokens}.npy + .json (keys, model revision, versions).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from chroma_common import (
    DEFAULT_MODEL, RULES_VERSION, TEXTS, bm25_tokenize, collection_name, docs, encode, load_model, load_records,
    model_revision, query_text, read_jsonl, source_sha, text_tag,
)
from chroma_eval_p1 import DEPTH, GOLDEN, KS, OUT_DIR, rank_of, rrf, summary

CACHE = OUT_DIR / "emb"


def doc_vectors(model_name: str, text: str, max_labels: int, max_tokens: int, ds: list[dict], sha: str,
                state: dict, batch: int, variant: str = "") -> np.ndarray:
    """Load cached document vectors, or embed and cache them. state holds the loaded model across calls.
    variant tells apart other texts built from the same records (e.g. category_eval's per-category texts)."""
    name = f"{collection_name(model_name, text, max_labels, sha)}-T{max_tokens}{variant}"
    npy, meta = CACHE / f"{name}.npy", CACHE / f"{name}.json"
    keys = [d["key"] for d in ds]
    if npy.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8"))["keys"] == keys:
        print(f"  cache {name}")
        return np.load(npy)
    model = model_for(model_name, max_tokens, state)
    started = time.perf_counter()
    vecs = np.asarray(encode(model, model_name, [d["embed"] for d in ds], "passage", batch), dtype=np.float32)
    seconds = time.perf_counter() - started
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    meta.write_text(json.dumps({
        "name": name, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": sha, "text": text, "rules": RULES_VERSION if text == "norm" else None,
        "max_labels": max_labels if text == "norm" else None, "variant": variant, "model": model_name,
        "model_revision": model_revision(model_name), "dim": int(vecs.shape[1]), "max_tokens": max_tokens,
        "device": str(model.device), "fp16": model.device.type == "cuda", "embed_seconds": round(seconds, 1),
        "versions": {p: version(p) for p in ("sentence-transformers", "torch", "transformers")},
        "keys": keys,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"  embedded {name} in {seconds:.0f}s")
    return vecs


def model_for(name: str, max_tokens: int, state: dict):
    if state.get("name") != name:  # one model in GPU memory at a time
        state.clear()
        state.update(name=name, model=load_model(name, max_tokens))
    state["model"].max_seq_length = max_tokens
    return state["model"]


def top_keys(scores: np.ndarray, keys: list[str]) -> list[list[str]]:
    """Row-wise top DEPTH by score (exact search)."""
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :DEPTH]  # stable: ties keep file order, like sorted()
    return [[keys[j] for j in row] for row in idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    p.add_argument("--texts", nargs="+", choices=TEXTS, default=list(TEXTS))
    p.add_argument("--max-labels", nargs="+", type=int, default=[60], help="norm only; 0 = never summarize")
    p.add_argument("--max-tokens", type=int, default=1024, help="document side truncation")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--show-changed", action="store_true")
    args = p.parse_args()

    items = read_jsonl(GOLDEN)
    gold = [Path(it["table"]).stem for it in items]  # tables.csv key, e.g. 101_DT_1B8000H
    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    variants = [("raw", 0)] * ("raw" in args.texts) + [("norm", n) for n in args.max_labels] * ("norm" in args.texts)
    runs: dict[str, list[int | None]] = {}
    tops: dict[str, list[str | None]] = {}
    state: dict = {}

    sparse_cache: dict[str, list[list[str]]] = {}
    for text, max_labels in variants:
        tag = text_tag(text, max_labels)
        ds = docs(records, text, max_labels)
        keys = [d["key"] for d in ds]
        queries = [query_text(it["question"], text) for it in items]
        if text not in sparse_cache:  # bm25_text keeps every label, so it does not depend on max_labels
            bm25 = BM25Okapi([tokenize(d["bm25"]) for d in ds])
            sparse_cache[text] = []
            for q in queries:
                s = np.asarray(bm25.get_scores(tokenize(q)))
                sparse_cache[text].append([keys[j] for j in np.argsort(-s, kind="stable")[:DEPTH] if s[j] > 0])
            lists = {f"bm25 {text}": sparse_cache[text]}
        else:
            lists = {}
        for model_name in args.models:
            print(f"{model_name} / {tag}")
            dv = doc_vectors(model_name, text, max_labels, args.max_tokens, ds, sha, state, args.batch)
            model = model_for(model_name, 512, state)
            qv = np.asarray(encode(model, model_name, queries, "query", args.batch), dtype=np.float32)
            dense = top_keys(qv @ dv.T, keys)  # unit vectors: dot product = cosine
            slug = model_name.split("/")[-1]
            lists[f"dense {slug} {tag}"] = dense
            lists[f"hybrid {slug} {tag}"] = [rrf(d, s) for d, s in zip(dense, sparse_cache[text])]
        for run, per_q in lists.items():
            runs[run] = [rank_of(g, top) for g, top in zip(gold, per_q)]
            tops[run] = [top[0] if top else None for top in per_q]

    kinds = {"answerable": [i for i, it in enumerate(items) if it["kind"] == "answerable"], "all": list(range(len(items)))}
    lines = [f"# Embedding comparison, Phase 1 golden (exact cosine, documents truncated at {args.max_tokens} tokens)", ""]
    for kind, idx in kinds.items():
        lines += [f"## {kind} (n={len(idx)})", "", "| run | Hit@1 | Hit@5 | Hit@10 | MRR@10 |", "| --- | --- | --- | --- | --- |"]
        for run in runs:
            s = summary([runs[run][i] for i in idx])
            lines.append(f"| {run} | " + " | ".join(f"{s[m]:.3f}" for m in (*(f"Hit@{k}" for k in KS), "MRR@10")) + " |")
        lines.append("")
    base = {run.rsplit(" ", 1)[0]: run for run in runs if run.endswith(" raw")}  # "dense bge-m3" -> its raw run
    moves = [(base[r.rsplit(" ", 1)[0]], r) for r in runs if not r.endswith(" raw") and r.rsplit(" ", 1)[0] in base]
    if moves:
        lines += ["## raw -> norm, rank moves (all questions)", ""]
        ahead = lambda x, y: (x or 999) < (y or 999)  # noqa: E731  (None = not in the top 50)
        for a, b in moves:
            better = sum(ahead(y, x) for x, y in zip(runs[a], runs[b]))
            worse = sum(ahead(x, y) for x, y in zip(runs[a], runs[b]))
            lines.append(f"- {b}: better {better}, worse {worse}, same {len(items) - better - worse}")
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "embed_eval_summary.md").write_text(report, encoding="utf-8")
    with (OUT_DIR / "embed_eval_ranks.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {"id": it["id"], "kind": it["kind"], "gold": gold[i], "question": it["question"]}
            row |= {run: runs[run][i] for run in runs} | {f"top1 {run}": tops[run][i] for run in tops}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {OUT_DIR / 'embed_eval_summary.md'}, {OUT_DIR / 'embed_eval_ranks.jsonl'}")

    if args.show_changed:
        for i, it in enumerate(items):
            moved = [f"{b} {runs[a][i]}->{runs[b][i]}" for a, b in moves if runs[a][i] != runs[b][i]]
            if moved:
                print(f"[{it['id']}] " + "  ".join(moved) + f"  Q: {it['question']}")


if __name__ == "__main__":
    main()
