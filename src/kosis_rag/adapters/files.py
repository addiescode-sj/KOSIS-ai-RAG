"""Files the search module reads (T2-41 S1): the table registry (from pipeline.py) and 시군구 -> 시도 (from vocab.py).

Only what the online search reads; build outputs (tables.jsonl, sigungu.json, Chroma builds, hq caches) are written
by the build code (kosis_rag_build from S6a).
"""

from __future__ import annotations

import json
from pathlib import Path

from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared.vocab import SIGUNGU


def read_records(path: Path) -> list[TableRecord]:
    """Read the table registry (data/tables.jsonl) the filters and the table texts are built from.

    In: path = a TableRecord jsonl file. Out: one TableRecord per non-empty line, in file order.
    Side effects: reads the file; raises on a missing file or an invalid line.
    Node: none directly (load_resources calls it once before the graph is built).
    """
    with path.open(encoding="utf-8") as f:
        return [TableRecord.model_validate_json(line) for line in f if line.strip()]


def load_sigungu(path: Path = SIGUNGU) -> dict[str, list[str]]:
    """{시군구 label: [시도, ...]} from data/sigungu.json; {} when the file is not there (the caller logs that once)."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
