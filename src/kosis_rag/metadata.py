"""Build data/tables.jsonl (T2-01): one TableRecord per provided table, for stage 2 search and period filters.

Listed facts (names, links, survey cycle) come from tables.csv, the catalog and the KOSIS table metadata.
Periods, items and classification labels come from the distinct values in the Parquet file, because the
metadata lists only part of the labels and writes some 5-year surveys with period code Y in the values.
Where the sources disagree the record keeps both and says so in `issues`; nothing is dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from kosis_ai.normalize import _meta_unit
from kosis_ai.schemas import MAX_DIMS, TableInfo
from kosis_rag.schemas import TableClass, TableRecord

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CATALOG = Path("reference") / "population_table_catalog.jsonl"


def _year(period: str) -> int | None:
    return int(period[:4]) if period[:4].isdigit() else None


def _distinct(table, column: str) -> list[str]:
    return sorted(v for v in table[column].unique().to_pylist() if v not in (None, ""))


def _values(path: Path) -> dict:
    """Distinct period codes, periods, items and (class name -> labels) from one Parquet file."""
    names = pq.read_schema(path).names
    dims = [i for i in range(1, MAX_DIMS + 1) if f"dim{i}_name" in names]
    columns = ["prd_se", "period", "item_name", *(f"dim{i}_{part}" for i in dims for part in ("name", "label"))]
    table = pq.read_table(path, columns=columns)
    classes: dict[str, set[str]] = {}
    for i in dims:
        pairs = table.group_by([f"dim{i}_name", f"dim{i}_label"]).aggregate([])
        for name, label in zip(pairs[f"dim{i}_name"].to_pylist(), pairs[f"dim{i}_label"].to_pylist()):
            if name:
                classes.setdefault(name, set()).update([label] if label else [])
    return {
        "prd_se": _distinct(table, "prd_se"),
        "periods": _distinct(table, "period"),
        "items": _distinct(table, "item_name"),
        "classes": [TableClass(name=n, labels=sorted(ls), n_labels=len(ls)) for n, ls in classes.items()],
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


def build_records(data_dir: Path = DATA_DIR) -> Iterator[TableRecord]:
    """One record per tables.csv row, in file order."""
    with (data_dir / CATALOG).open(encoding="utf-8") as f:
        catalog = {f"{c['org_id']}_{c['tbl_id']}": c for c in map(json.loads, f)}
    with (data_dir / "tables.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        info = TableInfo(**row)
        meta_path = data_dir / "table_metadata" / f"{info.key}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        cat = catalog.get(info.key)
        parquet = data_dir / info.parquet
        values = _values(parquet) if parquet.exists() else {}
        periods = values.get("periods", [])
        years = [y for y in map(_year, periods) if y is not None]
        record = TableRecord(
            info=info,
            table_name_en=(meta or {}).get("tblEngNm") or None,
            topic=(cat or {}).get("topic"),
            menu_paths=list(dict.fromkeys([info.menu_path, *(p["path"] for p in (cat or {}).get("paths", []))])),
            unit=_meta_unit(info.key, data_dir),
            cycle=[c for c in ((meta or {}).get("periodStr") or "").split("#") if c],
            contain_period=((meta or {}).get("containPeriod") or "").strip(),
            start=min(years, default=None),
            end=max(years, default=None),
            **values,
        )
        record.issues = _issues(record, cat, meta)
        record.search_text = _search_text(record)
        yield record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/tables.jsonl from the provided data (T2-01).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "tables.jsonl")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, count = time.perf_counter(), 0
    with args.out.open("w", encoding="utf-8") as f:
        for record in build_records(args.data_dir):
            f.write(record.model_dump_json() + "\n")
            count += 1
    print(f"{count} tables -> {args.out} ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
