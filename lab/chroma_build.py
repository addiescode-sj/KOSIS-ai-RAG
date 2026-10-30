"""Embed every table in data/tables.jsonl into the local ChromaDB folder (chroma/), plus a build manifest.

    python lab/chroma_build.py --text raw            # before: search_text as KOS-24 built it
    python lab/chroma_build.py --text norm           # after: KOS-88 embed_text (labels > 60 summarized)
    python lab/chroma_build.py --text norm --max-labels 100
    python lab/chroma_build.py --text norm --limit 50   # quick trial
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

from chroma_common import (
    CHROMA_DIR, DEFAULT_MODEL, MANIFEST_DIR, RULES_VERSION, TABLES, TEXTS,
    client, collection_name, docs, encode, load_model, load_records, metadata, model_revision, source_sha,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", choices=TEXTS, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=1024, help="model input is truncated here")
    p.add_argument("--max-labels", type=int, default=60, help="norm only: longer label lists are summarized")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only the first N tables (quick trial)")
    args = p.parse_args()

    sha = source_sha()
    records = load_records()[: args.limit]
    ds = docs(records, args.text, args.max_labels)
    texts = [d["embed"] for d in ds]
    lens = sorted(map(len, texts))
    print(f"{len(records)} tables, text={args.text}, chars median {lens[len(lens) // 2]} / max {lens[-1]}")

    model = load_model(args.model, args.max_tokens)
    print(f"model {args.model} on {model.device}, dim {model.get_sentence_embedding_dimension()}")
    started = time.perf_counter()
    vectors = encode(model, args.model, texts, "passage", args.batch)
    seconds = time.perf_counter() - started
    print(f"embedded in {seconds:.0f}s")

    name = collection_name(args.model, args.text, args.max_labels, sha) + ("-trial" if args.limit else "")
    store = client()
    if name in [c.name for c in store.list_collections()]:
        store.delete_collection(name)  # rebuild from scratch: the store is derived data
    col = store.create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"model": args.model, "text": args.text, "max_labels": args.max_labels, "source_sha": sha},
    )
    step = 500  # below get_max_batch_size() (5461)
    for i in range(0, len(ds), step):
        col.upsert(
            ids=[d["key"] for d in ds[i : i + step]],  # table key = document id, so a hit is the table itself
            documents=texts[i : i + step],
            embeddings=vectors[i : i + step].tolist(),
            metadatas=[metadata(r, d["fields"]) for r, d in zip(records[i : i + step], ds[i : i + step])],
        )

    manifest = {
        "collection": name,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(TABLES.relative_to(TABLES.parents[1])),
        "source_sha256": sha,
        "documents": col.count(),
        "text": args.text,
        "rules": RULES_VERSION if args.text == "norm" else None,
        "max_labels": args.max_labels if args.text == "norm" else None,
        "model": args.model,
        "model_revision": model_revision(args.model),
        "dim": int(vectors.shape[1]),
        "max_tokens": args.max_tokens,
        "device": str(model.device),
        "fp16": model.device.type == "cuda",
        "embed_seconds": round(seconds, 1),
        "versions": {pkg: version(pkg) for pkg in ("chromadb", "sentence-transformers", "torch", "transformers")}
        | {"python": platform.python_version()},
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"{name}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"collection {name}: {col.count()} documents in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
