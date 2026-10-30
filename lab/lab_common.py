"""Shared bits for the lab experiments: paths, the V0 search text, the embedding model, caches and metrics.

Gathered here on 10/30 from chroma_common, chroma_eval_p1, embed_eval, category_eval, retrieval_eval,
soft_route_eval and synth_router (those scripts were retired; the full versions are kept on the min branch).
Everything below behaves exactly as it did there, so cache file names and numbers do not change.

Text versions:
    raw   record["search_text"] as KOS-24 built it
    norm  normalize_text.build_doc(...)["embed_text"] (rules version + label cap go into the tag, e.g. n2-L60)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repo root (lab/ sits directly under it)
sys.path.insert(0, str(ROOT / "src"))  # kosis_rag.retriever.tokenize, the team's BM25 tokenizer

from normalize_text import RULES_VERSION, TABLES, build_doc, load_records, norm_query  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-m3"
TEXTS = ("raw", "norm")

OUT_DIR = ROOT / "data" / "lab"  # gitignored (/data/)
CACHE = OUT_DIR / "emb"  # document / question vectors (.npy)
SYNTH = OUT_DIR / "synth" / "synth_v0.jsonl"  # synthetic questions (plan 17)
ROUTING = ROOT / "eval" / "golden" / "routing_v0.jsonl"  # tune on dev only
RETRIEVAL = ROOT / "eval" / "golden" / "retrieval_golden.jsonl"  # main judging set
DEPTH, RRF_C, KS = 50, 60, (1, 5, 10)
METRICS = (*(f"Hit@{k}" for k in KS), "MRR@10")
B = 5000  # bootstrap resamples

# Some embedding models were trained with role prefixes; using the wrong (or no) prefix costs recall.
PREFIX = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


def source_sha(path: Path = TABLES) -> str:
    """sha256 of tables.jsonl; the first 8 hex chars go into the collection name, so a rebuilt file gets a new one."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_tag(text: str, max_labels: int) -> str:
    return "raw" if text == "raw" else f"{RULES_VERSION}-L{max_labels}"


def collection_name(model: str, text: str, max_labels: int, sha: str) -> str:
    """KOS-90 naming: {model}__{text rules}__{source sha8}, e.g. bge-m3__n2-L60__1a2b3c4d."""
    return f"{model.split('/')[-1]}__{text_tag(text, max_labels)}__{sha[:8]}"


def docs(records: list[dict], text: str, max_labels: int = 60) -> list[dict]:
    """Per table: key, embed (dense passage), bm25 (sparse text), fields. raw uses search_text for both texts,
    but the parsed fields either way, so both collections carry the same metadata filters."""
    out = []
    for r in records:
        d = build_doc(r, max_labels=max_labels)
        raw = text == "raw"
        out.append({"key": d["key"], "embed": r["search_text"] if raw else d["embed_text"],
                    "bm25": r["search_text"] if raw else d["bm25_text"], "fields": d["fields"]})
    return out


def query_text(question: str, text: str) -> str:
    """The query gets the same surface rules as the documents it is matched against."""
    return question if text == "raw" else norm_query(question)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(name: str, max_tokens: int):
    """sentence-transformers model on GPU in fp16 when there is one (bge-m3 fp16 peaks near 1.3GB on the 2070)."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
    if model.device.type == "cuda":
        model.half()
    model.max_seq_length = max_tokens
    return model


def encode(model, model_name: str, texts: list[str], role: str, batch: int = 16):
    """role 'query' or 'passage'. normalize -> unit vectors, so cosine similarity == dot product."""
    prefix = PREFIX.get(model_name, ("", ""))[0 if role == "query" else 1]
    return model.encode([prefix + t for t in texts], batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 100)


def model_revision(name: str) -> str | None:
    """Commit hash of the cached Hugging Face snapshot, so a later re-download can be told apart."""
    try:
        from huggingface_hub import try_to_load_from_cache

        parts = Path(try_to_load_from_cache(name, "config.json")).parts  # .../snapshots/<commit>/config.json
        return parts[parts.index("snapshots") + 1]
    except Exception:
        return None


def bm25_tokenize():
    """The team's BM25 tokenizer (words + Korean bigrams), so lab numbers match src/kosis_rag/retriever.py."""
    try:
        from kosis_rag.retriever import tokenize
    except ImportError:  # langchain-core missing in this env: same rule, copied
        import re

        word = re.compile(r"[0-9A-Za-z가-힣]+")

        def tokenize(text: str) -> list[str]:
            out = []
            for w in word.findall(text.lower()):
                out += [w, *(w[i : i + 2] for i in range(len(w) - 1))]
            return out
    return tokenize


# --- document / question vectors (cached) ---

def doc_vectors(model_name: str, text: str, max_labels: int, max_tokens: int, ds: list[dict], sha: str,
                state: dict, batch: int, variant: str = "") -> np.ndarray:
    """Load cached document vectors, or embed and cache them. state holds the loaded model across calls.
    variant tells apart other texts built from the same records."""
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


def question_vectors(model_name: str, texts: list[str], tag: str, state: dict, batch: int) -> np.ndarray:
    """Query-side vectors for many questions, cached like the document vectors (keyed by the texts' hash)."""
    sha = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:8]
    npy = CACHE / f"{model_name.split('/')[-1]}__q-{tag}__{sha}.npy"
    if npy.exists():
        print(f"  cache {npy.name}")
        return np.load(npy)
    vecs = np.asarray(encode(model_for(model_name, 512, state), model_name, texts, "query", batch), dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    print(f"  embedded {npy.name}")
    return vecs


# --- ranking and metrics ---

def top(scores: np.ndarray, keys: list[str], k: int = DEPTH) -> list[str]:
    return [keys[j] for j in np.argsort(-scores, kind="stable")[:k]]


def rrf(*lists: list[str], c: int = RRF_C) -> list[str]:
    score: dict[str, float] = {}
    for keys in lists:
        for rank, key in enumerate(keys, 1):
            score[key] = score.get(key, 0.0) + 1 / (c + rank)
    return sorted(score, key=score.get, reverse=True)


def rank_of(gold: str, keys: list[str]) -> int | None:
    return keys.index(gold) + 1 if gold in keys else None


def group_rank(groups: list[list[str]], keys: list[str]) -> int | None:
    """Worst over the groups of the best rank inside each group; None if some group is not in the list."""
    worst = 0
    for g in groups:
        ranks = [r for r in (rank_of(t, keys) for t in g) if r]
        if not ranks:
            return None
        worst = max(worst, min(ranks))
    return worst


def summary(ranks: list[int | None]) -> dict:
    n = max(len(ranks), 1)
    out = {f"Hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["MRR@10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    return out


def boot(diff: np.ndarray, groups: list[str] | None, rng) -> tuple[float, float]:
    """95% bootstrap interval of the mean difference; groups -> resample whole clusters (tables)."""
    if groups is None:
        d = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
    else:
        by: dict[str, list[float]] = {}
        for g, v in zip(groups, diff):
            by.setdefault(g, []).append(v)
        cl = [np.array(v) for v in by.values()]
        d = []
        for _ in range(B):
            pick = rng.integers(0, len(cl), len(cl))
            d.append(np.concatenate([cl[i] for i in pick]).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)
