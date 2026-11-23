"""T2-44 (KOS-138): copy Chroma collections into the pgvector store as they are — no re-embedding — and check them.

For each collection: read every row (id, document, metadata, embedding) from the Chroma server in pages, load it
through kosis_rag_build.vector_index.load (the same upsert every build uses; ids missing from Chroma are dropped, so
a rerun gives the same collection), keep the collection's metadata as its record, then check:
    count and id set equal · stored vectors equal (sample) · top-10 ids of sample queries equal (Chroma's HNSW is
    approximate, pgvector here is exact: a difference is printed, not a failure)

Needs the old Chroma server (e.g. the restored volume, `docker compose up -d chroma` on a checkout before T2-44, or
`docker run -d -p 8001:8000 -v kosis-ai-rag_chroma-data:/data chromadb/chroma:1.5.9`) and the new database
(`docker compose up -d postgres`). chromadb is not a runtime dependency any more; install the migrate group:

    uv sync --group migrate
    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/chroma_to_pgvector.py [--collection NAME ...]
        [--chroma-host localhost --chroma-port 8001] [--database-url postgresql://kosis:kosis@localhost:5432/kosis]
        [--list]

The database defaults to DATABASE_URL, else the compose postgres on localhost. Exit 1 when a count, id set or vector
check fails.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections.abc import Callable
from typing import Any

PAGE = 500  # rows per Chroma get
SAMPLE = 100  # vectors compared row by row
QUERIES = 20  # stored vectors used as queries for the top-10 check
TOP = 10


def read_all(collection: Any, page: int = PAGE) -> dict[str, list]:
    """Every row of a Chroma collection, in pages: {"ids", "documents", "metadatas", "embeddings"}."""
    out: dict[str, list] = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    offset = 0
    while True:
        got = collection.get(include=["documents", "metadatas", "embeddings"], limit=page, offset=offset)
        if not got["ids"]:
            return out
        out["ids"] += list(got["ids"])
        out["documents"] += [d if d is not None else "" for d in got["documents"]]
        out["metadatas"] += [dict(m or {}) for m in got["metadatas"]]
        out["embeddings"] += [[float(x) for x in e] for e in got["embeddings"]]
        offset += len(got["ids"])


def check(src: Any, dst: Any, rows: dict[str, list], seed: int = 0) -> tuple[list[str], list[str]]:
    """(problems, notes): problems fail the migration (counts, ids, vectors); notes are top-10 differences."""
    problems: list[str] = []
    notes: list[str] = []
    if dst.count() != len(rows["ids"]):
        problems.append(f"count: chroma {len(rows['ids'])}, pgvector {dst.count()}")
    stored = dst.get(include=[])["ids"]
    if set(stored) != set(rows["ids"]):
        problems.append(f"ids differ: {len(set(rows['ids']) - set(stored))} missing, "
                        f"{len(set(stored) - set(rows['ids']))} extra")
    rng = random.Random(seed)
    pick = rng.sample(range(len(rows["ids"])), min(SAMPLE, len(rows["ids"])))
    if pick:
        got = dst.get(ids=[rows["ids"][i] for i in pick], include=["embeddings", "documents", "metadatas"])
        by_id = {i: (d, m, e) for i, d, m, e in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])}
        worst, bad = 0.0, 0
        for i in pick:
            doc, meta, vec = by_id.get(rows["ids"][i], (None, None, None))
            if vec is None or doc != rows["documents"][i] or meta != rows["metadatas"][i]:
                bad += 1
                continue
            worst = max(worst, max(abs(float(a) - float(b)) for a, b in zip(vec, rows["embeddings"][i])))
        if bad:
            problems.append(f"{bad}/{len(pick)} sampled rows differ (document, metadata or missing)")
        if worst > 1e-6:
            problems.append(f"stored vectors differ by up to {worst:g} (float32 copy should be exact)")
    for i in rng.sample(range(len(rows["ids"])), min(QUERIES, len(rows["ids"]))):
        q = [rows["embeddings"][i]]
        n = min(TOP, len(rows["ids"]))
        a = src.query(query_embeddings=q, n_results=n, include=["distances"])["ids"][0]
        b = dst.query(query_embeddings=q, n_results=n, include=["distances"])["ids"][0]
        if list(a) != list(b):
            same = len(set(a) & set(b))
            notes.append(f"query {rows['ids'][i]}: top-{n} order differs ({same}/{n} ids shared)")
    return problems, notes


def migrate(src_client: Any, dst_client: Any, names: list[str], log: Callable[[str], None] = print) -> int:
    """Copy and check each collection; 0 when every check passes."""
    from kosis_rag_build.vector_index import load

    failed = 0
    for name in names:
        started = time.perf_counter()
        src = src_client.get_collection(name)
        rows = read_all(src)
        if not rows["ids"]:
            log(f"{name}: empty in chroma, skipped")
            continue
        record = dict(src.metadata or {})
        dims = {len(e) for e in rows["embeddings"]}
        if len(dims) != 1 or ("dim" in record and int(record["dim"]) not in dims):
            log(f"{name}: FAILED - vector lengths {sorted(dims)} vs record dim {record.get('dim')!r}")
            failed += 1
            continue
        record.setdefault("dim", dims.pop())
        count = load(dst_client, rows["ids"], rows["documents"], rows["metadatas"], rows["embeddings"], name, record)
        problems, notes = check(src, dst_client.get_collection(name), rows)
        status = "FAILED" if problems else "ok"
        log(f"{name}: {status} - {count} rows, dim {record['dim']}, {time.perf_counter() - started:.1f}s")
        for line in problems:
            log(f"  problem: {line}")
        for line in notes:
            log(f"  note: {line}")
        failed += bool(problems)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", action="append", default=None, help="repeat it (default: every collection)")
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--list", action="store_true", help="list the Chroma collections and their sizes, copy nothing")
    args = parser.parse_args(argv)

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("chromadb is not installed: run `uv sync --group migrate` first", file=sys.stderr)
        return 2
    src = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port,
                              settings=Settings(anonymized_telemetry=False))
    names = args.collection or sorted(c.name for c in src.list_collections())
    if args.list:
        for name in names:
            print(f"{name}  {src.get_collection(name).count()} rows")
        return 0

    from kosis_rag.adapters.vectorstore import safe_url, vector_client
    from kosis_rag.schemas.resources import VectorStoreConfig

    config = VectorStoreConfig(**({"url": args.database_url} if args.database_url else {}))
    print(f"chroma {args.chroma_host}:{args.chroma_port} -> {safe_url(config.url)}: {len(names)} collection(s)")
    code = migrate(src, vector_client(config), names)
    print("done" if code == 0 else "some collections FAILED (see above)")
    if code == 0:
        print("the search reads VECTOR_COLLECTION (was CHROMA_COLLECTION; the collection names are unchanged)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
