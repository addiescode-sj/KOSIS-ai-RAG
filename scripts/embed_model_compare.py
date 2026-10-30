"""T2-05: compare embedding models for the A build (one vector per table) on the T2-02 golden dev split.

Every model embeds the same documents (`index.to_document`) in memory and is scored with the T2-03 evaluator, as
`evals.run_eval --layer l1` does (no-answer threshold chosen on dev). Document vectors are cached under
data/embed_compare/ (gitignored), so a rerun only embeds the questions. The test split is not used.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/embed_model_compare.py [--models ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evals.evaluators import retrieval
from evals.run_eval import POOL, judged_keys
from evals.schemas import GOLDEN, load_items
from kosis_rag.index import REPO, read_records, to_document

OUT = REPO / "data" / "embed_compare"
# name -> (query prefix, document prefix); e5 was trained with these prefixes, KURE-v1 and bge-m3 use none
MODELS = {
    "nlpai-lab/KURE-v1": ("", ""),
    "BAAI/bge-m3": ("", ""),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


def embed_docs(model, name: str, docs: list[str], prefix: str) -> tuple[np.ndarray, dict]:
    path = OUT / (name.replace("/", "__") + ".npy")
    texts = [prefix + d for d in docs]
    tokens = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    info = {"max_seq_length": model.max_seq_length, "max_doc_tokens": max(tokens),
            "truncated_docs": sum(t > model.max_seq_length for t in tokens)}
    if path.exists():
        return np.load(path), info | {"embed_seconds": None}
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch
    started = time.perf_counter()
    vecs = model.encode([texts[i] for i in order], batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
    out = np.empty_like(vecs)
    out[order] = vecs
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out, info | {"embed_seconds": round(time.perf_counter() - started)}


def score(items, keys: list[str], q: np.ndarray, d: np.ndarray, k: int = 10) -> dict:
    sims = q @ d.T
    judged = judged_keys(POOL)
    rows = []
    for item, s in zip(items, sims):
        top = np.argsort(-s)[:k]
        row = {"qtype": item.qtype, "answerable": item.answerable, "top_score": float(s[top[0]])}
        if item.answerable:
            row |= retrieval.score_item(item, [keys[j] for j in top], judged.get(item.qid))
        rows.append(row)
    tau = retrieval.best_tau([(r["top_score"], not r["answerable"]) for r in rows])
    no_answer = retrieval.prf([(retrieval.predict_no_answer(r["top_score"], tau), not r["answerable"]) for r in rows])
    return {"metrics": retrieval.aggregate([r for r in rows if r["answerable"]]), "no_answer": no_answer, "tau": tau}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    records = read_records(REPO / "data" / "tables.jsonl")
    keys, docs = [r.info.key for r in records], [to_document(r) for r in records]
    items = [i for i in load_items(GOLDEN) if i.split == "dev"]
    results = {}
    for name in args.models:
        q_prefix, d_prefix = MODELS.get(name, ("", ""))
        model = SentenceTransformer(name, device=args.device)
        d, info = embed_docs(model, name, docs, d_prefix)
        started = time.perf_counter()
        q = model.encode([q_prefix + i.question for i in items], normalize_embeddings=True, convert_to_numpy=True)
        info["query_ms"] = round((time.perf_counter() - started) * 1000 / len(items), 1)
        results[name] = info | score(items, keys, q, d)
        a = results[name]["metrics"]["all"]
        print(f"{name:32s} n={a['n']} hit@1 {a['hit@1'] * a['n']:.0f} hit@5 {a['hit@5'] * a['n']:.0f} "
              f"hit@10 {a['hit@10'] * a['n']:.0f} MRR {a['rr']:.3f} | no-answer F1 {results[name]['no_answer']['f1']:.3f} "
              f"| {info}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
