"""Files the search module reads (T2-41 S1): the table registry (from pipeline.py), 시군구 -> 시도 (from vocab.py) and,
from S2, the rows of a question index build (from hq/index.py).

Only what the online search reads; build outputs (tables.jsonl, sigungu.json, Chroma builds, hq caches) are written
by the build code (kosis_rag_build from S6a).
"""

from __future__ import annotations

import json
from pathlib import Path

from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared.constants import DESC_SUFFIX
from kosis_rag.shared.vocab import REPO, SIGUNGU

TABLES = REPO / "data" / "tables.jsonl"  # the table registry (IndexSpec.tables None; was SearchSettings.tables)
HQ_DIR = REPO / "data" / "hq"  # question index builds (kosis_rag.hq.build writes them; the search only reads them)


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


def read_questions(build_id: str, root: Path = HQ_DIR) -> list[dict]:
    """The rows to index for a build: {"id", "key", "type", "text"} (T2-41 S2: from hq/index.py).

    In: build_id = a kosis_rag.hq.build id, or '<id>-desc' for its descriptions (one row per table, type
        'description'); root = data/hq. Out: rows in file order. Side effects: reads the build folder;
        FileNotFoundError when the build is missing.
    Node: none directly (load_resources and the index CLI call it; retrieve_hq_bm25 searches what it gives).
    """
    if build_id.endswith(DESC_SUFFIX):
        path = root / build_id[: -len(DESC_SUFFIX)] / "descriptions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [{"id": f"{r['key']}#d", "key": r["key"], "type": "description", "text": r["description"]} for r in rows]
    path = root / build_id / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
