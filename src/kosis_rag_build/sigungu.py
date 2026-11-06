"""data/sigungu.json: 시군구 label -> its 시도 names, read from the KOSIS region codes of one 읍면동 table (code[:2] is
the 시도). T2-41 S6a: from kosis_rag.shared.vocab (build_sigungu, main); the search reads the file through
kosis_rag.adapters.files.load_sigungu.

    python -m kosis_rag_build.cli sigungu --data-dir <provided data>     # graph/index_build.build_sigungu_graph
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kosis_rag.shared.vocab import SIDO_LABELS
from kosis_rag_build.schemas import SigunguBuildState

SIGUNGU_SOURCE = "values/101_DT_1IN1502.parquet"  # 인구, 가구 및 주택 - 읍면동: every 시도 and 시군구 with codes


def read_pairs(parquet: Path) -> list[dict[str, Any]]:
    """Distinct {dim1_code, dim1_label} rows of the table. Side effects: reads the Parquet file. Node: read_parquet."""
    import pyarrow.parquet as pq

    return pq.read_table(parquet, columns=["dim1_code", "dim1_label"]).group_by(["dim1_code", "dim1_label"]) \
        .aggregate([]).to_pylist()


def sigungu_of(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """{시군구 label: [시도, ...]} from region codes of 2 digits (시도) and 5 digits (시군구). Side effects: none.
    Node: build_sigungu."""
    sido = {r["dim1_code"]: SIDO_LABELS[r["dim1_label"]] for r in rows
            if len(r["dim1_code"] or "") == 2 and r["dim1_label"] in SIDO_LABELS}
    out: dict[str, set[str]] = {}
    for r in rows:
        code, label = r["dim1_code"] or "", r["dim1_label"]
        if len(code) == 5 and code[:2] in sido:
            out.setdefault(label, set()).add(sido[code[:2]])  # 중구 sits under several 시도
    return {label: sorted(names) for label, names in sorted(out.items())}


def build_sigungu(parquet: Path) -> dict[str, list[str]]:
    """{시군구 label: [시도, ...]} from a table whose region codes are 2 digits (시도) and 5 digits (시군구)."""
    return sigungu_of(read_pairs(parquet))


def dump(mapping: dict[str, list[str]]) -> str:
    """The file text (as the old vocab CLI wrote it: json indent=0, not ASCII-escaped)."""
    return json.dumps(mapping, ensure_ascii=False, indent=0)


# --- nodes (graph/index_build.build_sigungu_graph) ---

def _timed(name: str, trace: dict, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
    return {**trace, name: {"seconds": time.perf_counter() - started, **extra}}


def read_parquet(state: SigunguBuildState) -> dict:
    """In: state.parquet. Out: {"rows", "trace"}. Node: read_parquet."""
    started = time.perf_counter()
    return {"rows": read_pairs(state.parquet), "trace": _timed("read_parquet", state.trace, started)}


def build_sigungu_node(state: SigunguBuildState) -> dict:
    """In: state.rows. Out: {"mapping", "trace"}. Node: build_sigungu."""
    started = time.perf_counter()
    mapping = sigungu_of(state.rows)
    return {"mapping": mapping, "trace": _timed("build_sigungu", state.trace, started, sigungu=len(mapping))}


def write_json(state: SigunguBuildState) -> dict:
    """Write state.mapping to state.out (dump). Out: {"trace"}. Side effects: writes the file. Node: write_json."""
    started = time.perf_counter()
    state.out.write_text(dump(state.mapping), encoding="utf-8")
    return {"trace": _timed("write_json", state.trace, started)}
