"""Region chunks of the very large tables (T2-43, KOS-179): one region of a table = one chunk, values stay in Parquet.

The table index (chroma.py) holds one document per table and only counts the regions ("읍면동(4,484)"), so a
question that names a place cannot reach 101_DT_1B04005N (47.2M rows) or 101_DT_1B04006 (21.1M rows). Both tables
are region x age x item x period, and the only thing that differs between regions is which periods they really
have (dongs are created, merged and abolished). So the chunk unit is the region code:

    region_registry(parquet, info)  one group-by over the table -> list[RegionChunk]  (data/chunks/regions.jsonl)
    to_document / to_metadata       the Chroma document (no values) and its metadata (the Parquet filter)
    read_chunk_rows(parquet, meta)  chunk -> its rows, by the region code in the metadata

The chunks go to their own collection (CHUNK_PREFIX), never the table collection the search reads; its record
uses `chunk_format`, not `doc_format`, so `chroma --list` does not show it as a table build.

    python -m kosis_rag_build.cli chunks --data-dir <provided data> [--registry-only]
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from kosis_rag.adapters.models import EMBED_MODEL
from kosis_rag.schemas.common import TableInfo
from kosis_rag.shared import constants
from kosis_rag.shared.catalog import _META_FIELDS
from kosis_rag_build.chroma import clean_item

CHUNK_TABLES = ("101_DT_1B04005N", "101_DT_1B04006")  # the two largest tables (KOS-179)
CHUNK_FORMAT = "region1"  # bump when to_document or to_metadata changes: it is part of the collection name
CHUNK_PREFIX = "kosis_chunks__"
REGION_DIM = "dim1"  # both tables: dim1 = administrative region, dim2 = age
LEVELS = {2: "시도", 5: "시군구", 10: "읍면동"}  # by region code length; "00" is 전국
PRD_NAMES = {"M": "월", "Y": "년", "Q": "분기", "H": "반기"}
LABELS_SHOWN = 4  # labels of the other class shown in the document ("계, 0 - 4세, 5 - 9세, 10 - 14세 외 18개")

Level = Literal["전국", "시도", "시군구", "읍면동"]


class RegionChunk(BaseModel):
    """One region of one table. periods / missing: prd_se -> sorted periods the region has / the table's periods
    between the region's first and last that it lacks. rows = all rows of the region, value_rows = rows with a value."""

    table_key: str
    code: str
    label: str
    level: Level
    path: list[str]  # 시도 > 시군구 > 읍면동 labels, as far as the table has the parent codes
    parent: str | None  # the nearest parent code the table has
    periods: dict[str, list[str]]
    missing: dict[str, list[str]]
    rows: int
    value_rows: int
    other_class: str  # dim2 name, e.g. "5세별"
    other_labels: list[str]  # dim2 labels in code order
    items: list[str]

    @property
    def id(self) -> str:
        return f"{self.table_key}|{self.code}"


def level_of(code: str) -> Level:
    if code == "00":
        return "전국"
    if len(code) not in LEVELS or not code.isdigit():
        raise ValueError(f"region code {code!r}: expected 00 or a 2, 5 or 10 digit code")
    return LEVELS[len(code)]  # type: ignore[return-value]


def region_path(code: str, labels: dict[str, str]) -> tuple[list[str], str | None]:
    """(labels from the 시도 down to this region, nearest parent code). A parent the table does not list (old codes
    after a reorganization) is skipped, so the path can start below the 시도."""
    if code == "00":
        return [labels[code]], None
    parents = [p for p in (code[:2], code[:5]) if len(p) < len(code) and p in labels]
    return [labels[p] for p in parents] + [labels[code]], (parents[-1] if parents else None)


def missing_periods(have: list[str], table_periods: list[str]) -> list[str]:
    """Periods of the table between the region's first and last period that the region does not have."""
    if not have:
        return []
    got = set(have)
    return [p for p in table_periods if have[0] <= p <= have[-1] and p not in got]


def chunks_from_groups(info: TableInfo, groups: list[dict[str, Any]], other_class: str, other_labels: list[str],
                       items: list[str]) -> list[RegionChunk]:
    """groups = one dict per (region code, label, prd_se, period) with `rows` and `value_rows`. Raises when a code
    carries two labels (the document could then name the wrong place)."""
    labels: dict[str, str] = {}
    periods: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    rows: dict[str, int] = defaultdict(int)
    value_rows: dict[str, int] = defaultdict(int)
    table_periods: dict[str, set[str]] = defaultdict(set)
    for g in groups:
        code = g["code"]
        if labels.setdefault(code, g["label"]) != g["label"]:
            raise ValueError(f"{info.key}: region code {code} has two labels ({labels[code]!r}, {g['label']!r})")
        periods[code][g["prd_se"]].append(g["period"])
        table_periods[g["prd_se"]].add(g["period"])
        rows[code] += g["rows"]
        value_rows[code] += g["value_rows"]
    ordered = {se: sorted(ps) for se, ps in table_periods.items()}
    chunks = []
    for code in sorted(labels):
        path, parent = region_path(code, labels)
        have = {se: sorted(ps) for se, ps in sorted(periods[code].items())}
        gaps = {se: m for se, ps in have.items() if (m := missing_periods(ps, ordered[se]))}
        chunks.append(RegionChunk(
            table_key=info.key, code=code, label=labels[code], level=level_of(code), path=path, parent=parent,
            periods=have, missing=gaps, rows=rows[code], value_rows=value_rows[code], other_class=other_class,
            other_labels=other_labels, items=items))
    return chunks


def region_registry(parquet: Path, info: TableInfo) -> list[RegionChunk]:
    """One pass over the table's Parquet: a group-by on (region, period), never one read per region.
    Raises when (region, period, dim2, item) repeats, so the chunks cover every row exactly once."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = [f"{REGION_DIM}_code", f"{REGION_DIM}_label", "prd_se", "period", "dim2_name", "dim2_code", "dim2_label",
            "item_id", "item_name", "value_kind"]
    table = pq.read_table(parquet, columns=cols, read_dictionary=cols).unify_dictionaries().combine_chunks()
    keys = table.group_by([f"{REGION_DIM}_code", "period", "dim2_code", "item_id"]).aggregate([([], "count_all")])
    if keys.num_rows != table.num_rows:
        raise ValueError(f"{info.key}: {table.num_rows - keys.num_rows} rows repeat a (region, period, dim2, item) key")
    table = table.append_column("has_value", pc.cast(pc.equal(table["value_kind"], "ok"), "int64"))
    grouped = table.group_by([f"{REGION_DIM}_code", f"{REGION_DIM}_label", "prd_se", "period"]).aggregate(
        [([], "count_all"), ("has_value", "sum")])
    groups = [{"code": g[f"{REGION_DIM}_code"], "label": g[f"{REGION_DIM}_label"], "prd_se": g["prd_se"],
               "period": g["period"], "rows": g["count_all"], "value_rows": g["has_value_sum"]}
              for g in grouped.to_pylist()]
    other = table.group_by(["dim2_name", "dim2_code", "dim2_label"]).aggregate([]).to_pylist()
    other.sort(key=lambda o: (len(o["dim2_code"]), o["dim2_code"]))
    items = table.group_by(["item_id", "item_name"]).aggregate([]).to_pylist()
    return chunks_from_groups(info, groups, other[0]["dim2_name"], [o["dim2_label"] for o in other],
                              [i["item_name"] for i in sorted(items, key=lambda i: i["item_id"])])


def _span(periods: list[str]) -> str:
    return f"{periods[0]}~{periods[-1]}"


def to_document(chunk: RegionChunk, info: TableInfo) -> str:
    """The embedding text: the region path first, then what the table is and which periods this region really has.
    No values: they are read from Parquet (read_chunk_rows)."""
    cleaned = [clean_item(i) for i in chunk.items]
    units = list(dict.fromkeys(u for _, u in cleaned if u))
    spans = ", ".join(f"{PRD_NAMES.get(se, se)} {_span(ps)}" for se, ps in chunk.periods.items())
    gaps = sum(len(m) for m in chunk.missing.values())
    shown = chunk.other_labels[:LABELS_SHOWN]
    more = f" 외 {len(chunk.other_labels) - len(shown)}개" if len(chunk.other_labels) > len(shown) else ""
    parts = [
        " ".join(chunk.path),
        info.table_name,
        f"지역: {' > '.join(chunk.path)} ({chunk.level})",
        f"수록기간: {spans}" + (f" (중간에 빈 시점 {gaps}개)" if gaps else ""),
        f"{chunk.other_class}: {', '.join(shown)}{more}",
        "항목: " + ", ".join(name for name, _ in cleaned) + (f" ({', '.join(units)})" if units else ""),
        f"조사명: {info.survey_name}",
    ]
    return " | ".join(parts)


def to_metadata(chunk: RegionChunk, info: TableInfo) -> dict:
    """Chroma metadata: the table's listing fields (shared.catalog._META_FIELDS, so to_candidate works on a chunk),
    then the region and, per cycle, first / last period and count as integers for range filters. Missing periods
    are listed only when there are any (Chroma rejects empty lists)."""
    meta: dict[str, Any] = {name: getattr(info, name) for name in _META_FIELDS}
    meta |= {"part": "region", "region_code": chunk.code, "region_label": chunk.label, "region_level": chunk.level,
             "region_path": " > ".join(chunk.path), "parent_code": chunk.parent, "rows": chunk.rows,
             "value_rows": chunk.value_rows, "prd_se": "|".join(chunk.periods)}
    for se, ps in chunk.periods.items():
        s = se.lower()
        meta |= {f"start_{s}": int(ps[0]), f"end_{s}": int(ps[-1]), f"n_{s}": len(ps)}
        if chunk.missing.get(se):
            meta[f"missing_{s}"] = chunk.missing[se]
    return {k: v for k, v in meta.items() if v is not None}


def read_chunk_rows(parquet: Path, meta: dict):
    """The rows of one chunk as a pyarrow Table: the Parquet filter is the region code in the chunk's metadata."""
    import pyarrow.parquet as pq

    return pq.read_table(parquet, filters=[(f"{REGION_DIM}_code", "=", meta["region_code"])])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def collection_name(data_sha256: str) -> str:
    """Same Parquet files and format -> same name; new monthly data -> a new collection (the old one stays)."""
    name = f"{CHUNK_PREFIX}{CHUNK_FORMAT}__{hashlib.sha256(f'{data_sha256}|{CHUNK_FORMAT}'.encode()).hexdigest()[:8]}"
    if not constants.COLLECTION_NAME.match(name):
        raise ValueError(f"{name!r} is not a valid Chroma collection name")
    return name


def chunk_record(index_version: str, data_sha256: str, chunks: list[RegionChunk], dim: int, max_tokens: int) -> dict:
    """Collection metadata: what the chunks were built from. No `doc_format` key on purpose (see the module docstring)."""
    per_table: dict[str, int] = defaultdict(int)
    for c in chunks:
        per_table[c.table_key] += 1
    return {"index_version": index_version, "data_sha256": data_sha256, "chunk_format": CHUNK_FORMAT,
            "chunk_unit": "table x region code", "tables": "|".join(sorted(per_table)),
            "embed_model": EMBED_MODEL, "dim": dim, "normalized": True, "hnsw:space": "cosine",
            "n_docs": len(chunks), "n_rows": sum(c.rows for c in chunks), "max_doc_tokens": max_tokens}


def write_registry(path: Path, chunks: list[RegionChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")


def read_registry(path: Path) -> list[RegionChunk]:
    with path.open(encoding="utf-8") as f:
        return [RegionChunk.model_validate_json(line) for line in f if line.strip()]
