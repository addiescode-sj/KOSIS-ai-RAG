"""Build data/tables.jsonl (T2-01): one TableRecord per provided table, for stage 2 search and period filters.

Listed facts (names, links, survey cycle) come from tables.csv, the catalog and the KOSIS table metadata.
Periods, items and classification labels come from the distinct values in the Parquet file, because the
metadata lists only part of the labels and writes some 5-year surveys with period code Y in the values.
Where the sources disagree the record keeps both and says so in `issues`; nothing is dropped.

T2-41 S6a (was metadata.py of the search package): the graph is graph/index_build.build_tables_graph over the nodes below
(read_parquet -> build_records -> write_jsonl); `python -m kosis_rag_build.cli tables` runs it. build_records(data_dir)
gives the same records one by one (read_table + to_record per tables.csv row).
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from kosis_rag.schemas.catalog import TableClass, TableRecord
from kosis_rag.schemas.common import MAX_DIMS, TableInfo
from kosis_rag.shared.units import meta_unit
from kosis_rag.shared.vocab import REGION_LEVELS
from kosis_rag_build.schemas import TablesBuildState

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # the repo's data/ (src/kosis_rag_build -> repo)
CATALOG = Path("reference") / "population_table_catalog.jsonl"
# Region classes only (not e.g. 성씨·본관, where 동부 is a clan seat): their totals and 시도 go first in the label list
REGION_CLASS = re.compile(r"행정구역|시도|시군구|구시군|읍면동|지역")
NATION = {"전국", "계", "총계", "합계"}
URBAN = {"동부", "읍부", "면부"}
SIDO = {
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도",
    "강원도", "강원특별자치도", "충청북도", "충청남도", "전라북도", "전북특별자치도", "전라남도", "경상북도", "경상남도",
    "제주특별자치도", "제주도",
}
TOP_REGIONS = NATION | URBAN | SIDO


def region_level(code: str, label: str, class_name: str = "") -> str:
    """Level from the label first (works for flat codes), then the KOSIS code length, then the class name."""
    if label in NATION:
        return "전국"
    if label in URBAN:
        return "동읍면부"
    if label in SIDO:
        return "시도"
    n = len(code or "")
    if n == 5:
        return "시군구"
    if 7 <= n <= 10:
        return "읍면동"
    # flat codes (e.g. 법무부 tables): the class name says what the lower level is
    if "읍면동" in class_name:
        return "읍면동"
    if re.search(r"시군구|구시군", class_name):
        return "시군구"
    return "하위 지역"


def _region_levels(name: str, pairs: list[tuple[str, str]]) -> dict[str, int]:
    """{level: label count} for a region class; empty unless it lists 시도 (국적(지역)별 lists countries)."""
    if not REGION_CLASS.search(name):
        return {}
    found: dict[str, set[str]] = {}
    for code, label in pairs:
        found.setdefault(region_level(code, label, name), set()).add(label)
    if len(found.get("시도", ())) < 3:
        return {}
    return {lv: len(found[lv]) for lv in REGION_LEVELS if lv in found}


def _year(period: str) -> int | None:
    return int(period[:4]) if period[:4].isdigit() else None


def _distinct(values: set) -> list[str]:
    return sorted(v for v in values if v not in (None, ""))


BATCH_ROWS = 500_000  # rows per Parquet batch; bounds memory to a few hundred MB whatever the table size


def _values(path: Path) -> dict:
    """Distinct period codes, periods, items and (class name -> labels) from one Parquet file.

    Read in row batches, not whole columns: the distinct values and (name, code, label) pairs are small, the
    columns are not (the largest table has 111 million rows; its string columns loaded whole need over 16GB,
    which froze a 16GB PC). Same result as the whole-column read; only labels that share one code (a missing
    code) now come out in label order, where that read left their order to the group-by.
    """
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    dims = [i for i in range(1, MAX_DIMS + 1) if f"dim{i}_name" in names]
    columns = ["prd_se", "period", "item_name", *(f"dim{i}_{part}" for i in dims for part in ("name", "code", "label"))]
    distinct: dict[str, set] = {c: set() for c in ("prd_se", "period", "item_name")}
    seen: dict[int, set[tuple]] = {i: set() for i in dims}  # dim index -> {(name, code, label)}
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=columns):
        for column in distinct:
            distinct[column].update(batch[column].unique().to_pylist())
        for i in dims:
            cols = [f"dim{i}_name", f"dim{i}_code", f"dim{i}_label"]
            pairs = pa.Table.from_batches([batch]).select(cols).group_by(cols).aggregate([])
            seen[i].update(zip(*(pairs[c].to_pylist() for c in cols)))
    coded: dict[str, list[tuple[str, str]]] = {}
    for i in dims:
        for name, code, label in sorted(seen[i], key=lambda t: tuple(x or "" for x in t)):
            if name and label:
                coded.setdefault(name, []).append((code or "", label))
    classes = []
    for name, pairs in coded.items():
        # KOSIS code order puts higher levels first (00 전국, 11 서울특별시, 11010 종로구 ...), so a capped label list
        # keeps 전국 and the 시도; the same label under two codes (중구) is kept once, at its first place
        labels = list(dict.fromkeys(label for _, label in sorted(pairs, key=lambda p: (len(p[0]), p[0]))))
        if REGION_CLASS.search(name):
            # Some tables (e.g. 법무부 등록외국인) use flat codes that interleave 시도 with their 시군구; put the
            # national total and the 시도 first anyway, keeping their order
            labels = [x for x in labels if x in TOP_REGIONS] + [x for x in labels if x not in TOP_REGIONS]
        classes.append(TableClass(name=name, labels=labels, n_labels=len(labels), levels=_region_levels(name, pairs)))
    return {
        "prd_se": _distinct(distinct["prd_se"]),
        "periods": _distinct(distinct["period"]),
        "items": _distinct(distinct["item_name"]),
        "classes": classes,
    }


def _listed_years(catalog: dict | None) -> tuple[int | None, int | None]:
    years = [y for p in (catalog or {}).get("catalog_periods", []) for y in (_year(p["start"]), _year(p["end"])) if y]
    return (min(years), max(years)) if years else (None, None)


def _issues(record: TableRecord, catalog: dict | None, meta: dict | None) -> list[str]:
    issues = [f"no_{name}" for name, src in (("catalog", catalog), ("metadata", meta)) if src is None]
    if not record.periods:
        return [*issues, "no_values"]
    listed_start, listed_end = _listed_years(catalog)
    if listed_start is not None and listed_start != record.start:
        issues.append(f"start_differs:{listed_start}!={record.start}")
    if listed_end is not None and listed_end != record.end:
        issues.append(f"end_differs:{listed_end}!={record.end}")
    # A one-period table (most 5-year census tables) cannot show its cycle in the values; only flag the rest.
    if record.cycle and set(record.cycle) != set(record.prd_se) and len(record.periods) > 1:
        issues.append(f"cycle_code_differs:{'#'.join(record.cycle)}!={'#'.join(record.prd_se)}")
    return issues


def _search_text(record: TableRecord) -> str:
    info = record.info
    parts = [info.table_name, record.table_name_en, info.survey_name, info.organization, record.topic,
             *record.menu_paths, *record.items, *(c.name for c in record.classes),
             *(label for c in record.classes for label in c.labels)]
    return " ".join(dict.fromkeys(p for p in parts if p))


def read_catalog(data_dir: Path) -> dict[str, dict]:
    """The population table catalog by table key (reference/population_table_catalog.jsonl)."""
    with (data_dir / CATALOG).open(encoding="utf-8") as f:
        return {f"{c['org_id']}_{c['tbl_id']}": c for c in map(json.loads, f)}


def read_rows(data_dir: Path) -> list[dict[str, str]]:
    """The rows of tables.csv, in file order."""
    with (data_dir / "tables.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_table(data_dir: Path, row: dict[str, str], catalog: dict[str, dict]) -> dict[str, Any]:
    """What one table's record is made from: {info, meta (KOSIS metadata or None), cat (catalog entry or None),
    values (distinct values of its Parquet file, {} when there is none), unit}.

    Side effects: reads the table's metadata JSON and Parquet file. Node: read_parquet.
    """
    info = TableInfo(**row)
    meta_path = data_dir / "table_metadata" / f"{info.key}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    parquet = data_dir / info.parquet
    return {"info": info, "meta": meta, "cat": catalog.get(info.key),
            "values": _values(parquet) if parquet.exists() else {}, "unit": meta_unit(info.key, data_dir)}


def to_record(source: dict[str, Any]) -> TableRecord:
    """The TableRecord of one read_table result (issues and search_text filled). Side effects: none.
    Node: build_records."""
    info, meta, cat, values = source["info"], source["meta"], source["cat"], source["values"]
    periods = values.get("periods", [])
    years = [y for y in map(_year, periods) if y is not None]
    record = TableRecord(
        info=info,
        table_name_en=(meta or {}).get("tblEngNm") or None,
        topic=(cat or {}).get("topic"),
        menu_paths=list(dict.fromkeys([info.menu_path, *(p["path"] for p in (cat or {}).get("paths", []))])),
        unit=source["unit"],
        cycle=[c for c in ((meta or {}).get("periodStr") or "").split("#") if c],
        contain_period=((meta or {}).get("containPeriod") or "").strip(),
        start=min(years, default=None),
        end=max(years, default=None),
        **values,
    )
    record.issues = _issues(record, cat, meta)
    record.search_text = _search_text(record)
    return record


def build_records(data_dir: Path = DATA_DIR) -> Iterator[TableRecord]:
    """One record per tables.csv row, in file order."""
    catalog = read_catalog(data_dir)
    for row in read_rows(data_dir):
        yield to_record(read_table(data_dir, row, catalog))


# --- nodes (graph/index_build.build_tables_graph) ---

def _timed(name: str, trace: dict, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
    return {**trace, name: {"seconds": time.perf_counter() - started, **extra}}


def read_parquet(state: TablesBuildState) -> dict:
    """Every table's sources (read_table per tables.csv row).

    In: state.data_dir. Out: {"sources", "trace"}. Side effects: reads the provided data. Node: read_parquet.
    """
    started = time.perf_counter()
    catalog = read_catalog(state.data_dir)
    sources = [read_table(state.data_dir, row, catalog) for row in read_rows(state.data_dir)]
    return {"sources": sources, "trace": _timed("read_parquet", state.trace, started, tables=len(sources))}


def build_records_node(state: TablesBuildState) -> dict:
    """In: state.sources. Out: {"records" (to_record of each, in order), "trace"}. Side effects: none.
    Node: build_records."""
    started = time.perf_counter()
    return {"records": [to_record(s) for s in state.sources],
            "trace": _timed("build_records", state.trace, started)}


def write_jsonl(state: TablesBuildState) -> dict:
    """Write one record per line (model_dump_json) to state.out, creating its folder.

    Out: {"count", "trace"}. Side effects: writes the file. Node: write_jsonl.
    """
    started = time.perf_counter()
    state.out.parent.mkdir(parents=True, exist_ok=True)
    with state.out.open("w", encoding="utf-8") as f:
        for record in state.records:
            f.write(record.model_dump_json() + "\n")
    return {"count": len(state.records), "trace": _timed("write_jsonl", state.trace, started)}
