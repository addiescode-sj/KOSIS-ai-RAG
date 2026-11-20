"""T2-30 (KOS-101): check a loaded build against data/tables.jsonl — count, field fill, the mentor's where query.

Reads the collection in the vector store (DATABASE_URL, default the compose postgres; no embedding). T2-44: the
mentor's $and / $or / $contains query is decided here on the loaded metadata (the store takes only `key $in`). With --snapshot it also writes {id: metadata} to a JSON
file, so two runs around a reload can be compared for the same result.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/check_build.py --collection <name> [--snapshot out.json]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kosis_rag.adapters.vectorstore import vector_client
from kosis_rag.adapters.files import read_records
from kosis_rag.schemas.resources import VectorStoreConfig
from kosis_rag_build.vector_index import to_metadata

REPO = Path(__file__).resolve().parents[1]

FIELDS = ("f_years", "f_cycle", "f_regions", "f_region_level", "f_measure", "f_unit", "f_dims")
# mentor example: tables with 서울특별시 rows, 2025 values and a region breakdown
MENTOR = {"$and": [{"f_regions": {"$contains": "서울특별시"}}, {"f_years": {"$contains": 2025}},
                   {"f_dims": {"$contains": "행정구역"}}]}
MENTOR_ANY_REGION = {"$and": [{"f_regions": {"$contains": "서울특별시"}}, {"f_years": {"$contains": 2025}},
                              {"$or": [{"f_dims": {"$contains": k}} for k in ("행정구역", "시도", "시군구", "읍면동")]}]}


def matches(meta: dict, where: dict) -> bool:
    """Chroma's $and / $or / $contains on one row's metadata (list fields hold the value)."""
    if "$and" in where:
        return all(matches(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(matches(meta, w) for w in where["$or"])
    (field, cond), = where.items()
    return cond["$contains"] in (meta.get(field) or [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", required=True)
    parser.add_argument("--tables", type=Path, default=REPO / "data" / "tables.jsonl")
    parser.add_argument("--snapshot", type=Path, default=None)
    args = parser.parse_args(argv)

    expected = {r.info.key: to_metadata(r) for r in read_records(args.tables)}
    url = os.environ.get("DATABASE_URL")
    config = VectorStoreConfig(**({"url": url} if url else {}))
    collection = vector_client(config).get_collection(args.collection, embedding_function=None)
    got = collection.get(include=["metadatas"])
    loaded = dict(zip(got["ids"], got["metadatas"]))
    n = len(expected)
    print(f"documents {len(loaded)} / {n}, missing {len(set(expected) - set(loaded))}, extra {len(set(loaded) - set(expected))}")
    print(f"metadata equal to to_metadata(): {sum(loaded.get(k) == m for k, m in expected.items())} / {n}")
    for field in FIELDS:
        filled = sum(field in m for m in loaded.values())
        print(f"  {field:15s} {filled:5d}  {filled / n:.1%}")
    print(f"  {'f_national':15s} {sum(m.get('f_national') is True for m in loaded.values()):5d}")
    for label, where in (("mentor (행정구역)", MENTOR), ("mentor (any region breakdown)", MENTOR_ANY_REGION)):
        print(f"{label}: {sum(matches(m, where) for m in loaded.values())} tables")
    print(f"collection record: {collection.metadata}")
    if args.snapshot:
        args.snapshot.write_text(json.dumps(loaded, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        print(f"snapshot -> {args.snapshot}")
    return 0 if len(loaded) == n and set(loaded) == set(expected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
