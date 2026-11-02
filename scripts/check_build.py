"""T2-30 (KOS-101): check a loaded build against data/tables.jsonl — count, field fill, the mentor's where query.

Reads the collection on the shared server (no embedding). With --snapshot it also writes {id: metadata} to a JSON
file, so two runs around a reload can be compared for the same result.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/check_build.py --collection <name> [--snapshot out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kosis_rag.index import REPO, read_records, to_metadata
from kosis_rag.vectorstore import chroma_client

FIELDS = ("f_years", "f_cycle", "f_regions", "f_region_level", "f_measure", "f_unit", "f_dims")
# mentor example: tables with 서울특별시 rows, 2025 values and a region breakdown
MENTOR = {"$and": [{"f_regions": {"$contains": "서울특별시"}}, {"f_years": {"$contains": 2025}},
                   {"f_dims": {"$contains": "행정구역"}}]}
MENTOR_ANY_REGION = {"$and": [{"f_regions": {"$contains": "서울특별시"}}, {"f_years": {"$contains": 2025}},
                              {"$or": [{"f_dims": {"$contains": k}} for k in ("행정구역", "시도", "시군구", "읍면동")]}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", required=True)
    parser.add_argument("--tables", type=Path, default=REPO / "data" / "tables.jsonl")
    parser.add_argument("--snapshot", type=Path, default=None)
    args = parser.parse_args(argv)

    expected = {r.info.key: to_metadata(r) for r in read_records(args.tables)}
    collection = chroma_client().get_collection(args.collection, embedding_function=None)
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
        print(f"{label}: {len(collection.get(where=where, include=[])['ids'])} tables")
    print(f"collection record: {collection.metadata}")
    if args.snapshot:
        args.snapshot.write_text(json.dumps(loaded, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        print(f"snapshot -> {args.snapshot}")
    return 0 if len(loaded) == n and set(loaded) == set(expected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
