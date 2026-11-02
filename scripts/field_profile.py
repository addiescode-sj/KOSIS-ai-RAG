"""T2-30 (KOS-96): profile the per-table fields over data/tables.jsonl and write docs/reports/t2-30-fields.md.

Only aggregates and catalog tables go in the report; no golden question or answer key (T2-02 rule).

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/field_profile.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from kosis_rag.fields import ITEM_UNITS, LAST_BRACKET, TableFields, item_unit, table_fields
from kosis_rag.index import REPO, read_records

OUT = REPO / "docs" / "reports" / "t2-30-fields.md"
EXAMPLE = "101_DT_1PB1607"  # mentor example: 이동자유형별 인구, list A11_2015_1_10
WORDS = ("출생", "혼인")  # KOS-92: words used both as the counted thing and as a breakdown


def _value_counts(fields: list[TableFields], name: str) -> Counter:
    counts: Counter = Counter()
    for f in fields:
        value = getattr(f, name)
        if isinstance(value, list):
            counts.update(str(v) for v in value)
        elif value is not None:
            counts[str(value)] += 1
    return counts


def _show(value) -> str:
    if isinstance(value, list) and len(value) > 6 and all(isinstance(v, int) for v in value):
        return f"{value[0]}~{value[-1]} ({len(value)}개 해)"
    if isinstance(value, list) and len(value) > 6:
        return f"{len(value)}개 ({', '.join(map(str, value[:3]))} …)"
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def _filled(value) -> bool:
    return value is not None and value != []


def _unit_sources(records) -> Counter:
    out: Counter = Counter()
    for r in records:
        if r.unit:
            out["표 단위"] += 1
            continue
        units = [m.group(1).strip() if (m := LAST_BRACKET.search(i)) else None for i in r.items]
        known = {u for u in units if u in ITEM_UNITS}
        if item_unit(r.items):
            out["항목 이름 (모든 항목 같은 단위)"] += 1
        elif known and len(known) > 1 and all(u in ITEM_UNITS for u in units):
            out["없음: 항목마다 단위가 다름"] += 1
        elif known:
            out["없음: 일부 항목에만 단위"] += 1
        else:
            out["없음: 단위 정보 없음"] += 1
    return out


def report(records) -> str:
    fields = [table_fields(r) for r in records]
    n = len(fields)
    names = [k for k in TableFields.model_fields if k != "key"]
    lines = ["# T2-30 표별 메타데이터 필드 프로파일 (KOS-96)", "",
             f"> 대상 `data/tables.jsonl` {n:,}표 · 생성 `scripts/field_profile.py` · 필드 정의 `src/kosis_rag/fields.py`", "",
             "## 1. 필드별 채움률·고유값", "", "| 필드 | 채움 | 채움률 | 고유값 수 | 상위 값 |", "| --- | ---: | ---: | ---: | --- |"]
    for name in names:
        filled = sum(_filled(getattr(f, name)) for f in fields)
        counts = _value_counts(fields, name)
        top = ", ".join(f"{v} {c:,}" for v, c in counts.most_common(8 if name != "available_years" else 0))
        if name == "available_years":
            years = sorted(int(y) for y in counts)
            top = f"{years[0]}~{years[-1]}"
        lines.append(f"| `{name}` | {filled:,} | {filled / n:.1%} | {len(counts):,} | {top} |")
    national = sum(f.regions is None for f in fields)
    lines += ["", f"- `regions`가 None인 표 {national:,}개는 행정구역 분류가 없는 표(전국 단위)라 \"해당 없음\"이다. "
              f"이를 빼면 시도 이름이 있는 표 {n - national:,}개 모두 채워져 있다.",
              f"- `region_level`이 빈 표 {sum(not f.region_level for f in fields):,}개 중 {sum(not f.region_level and f.regions is None for f in fields):,}개가 같은 전국 표다. "
              "나머지는 시도 이름이 행정구역이 아닌 분류(현거주지별·5년전 거주지별·출생지별·통근통학지 등)에만 있는 표다. "
              "이런 표도 `regions`는 채워지므로, 출생지·이전 거주지를 지역 조건으로 볼지는 KOS-97에서 정한다.",
              "- `measure`는 표 이름이 \"A/B/C별 X\" 꼴이 아닌 표에서 비워 둔다(T2-27).", "",
              "## 2. 단위 (`measure_unit`)", "",
              "표 단위가 없으면 항목 이름 끝 괄호를 아래 목록으로 읽고, **모든 항목이 같은 단위일 때만** 채운다. "
              "괄호 없는 항목이 %·금액일 수 있어 일부 항목에만 단위가 있는 표는 비워 둔다.", "",
              f"- 인정 단위: {' · '.join(sorted(ITEM_UNITS))}", "", "| 출처 | 표 수 |", "| --- | ---: |"]
    for source, count in _unit_sources(records).most_common():
        lines.append(f"| {source} | {count:,} |")
    others = Counter(m.group(1).strip() for r in records if not r.unit for i in r.items
                     if (m := LAST_BRACKET.search(i)) and m.group(1).strip() not in ITEM_UNITS)
    lines += ["", "단위로 보지 않은 끝 괄호 상위 값(KOS-97 단위 사전에서 다시 본다): "
              + ", ".join(f"{v} {c}" for v, c in others.most_common(12)), ""]

    lines += ["## 3. KOS-92 확인 — 같은 단어가 세는 대상과 분류 기준에 섞인 표", "",
              "| 단어 | 표 이름에 든 표 | 세는 대상(`measure`)에 있음 | 분류(`dimensions`)에만 있음 | 둘 다 없음 | (measure, dimensions) 조합 수 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for word in WORDS:
        group = [f for r, f in zip(records, fields) if word in r.info.table_name]
        counted = sum(bool(f.measure and word in f.measure) for f in group)
        dim_only = sum(not (f.measure and word in f.measure) and any(word in d for d in f.dimensions) for f in group)
        combos = len({(f.measure, tuple(f.dimensions)) for f in group})
        lines.append(f"| {word} | {len(group)} | {counted} | {dim_only} | {len(group) - counted - dim_only} | {combos} |")
    lines += ["", "세는 대상 표와 분류 역할 표가 `measure` 한 필드로 갈린다. \"둘 다 없음\"은 표 이름 꼴이 맞지 않아 "
              "measure가 비고 분류 이름에도 그 단어가 없는 표다.", ""]

    example = next((f for f in fields if f.key == EXAMPLE), None)
    if example:
        record = next(r for r in records if r.info.key == EXAMPLE)
        lines += ["## 4. 멘토 예시 표 — 이동자유형별 인구 (목록 `A11_2015_1_10`)", "",
                  f"`{EXAMPLE}` {record.info.table_name}", "", "| 필드 | 값 |", "| --- | --- |",
                  *(f"| `{k}` | {_show(v)} |" for k, v in example.model_dump().items()), "",
                  "- `dimensions`의 \"현 거주지 성\"은 표 이름의 \"현 거주지 성, 연령 및 …\"을 쉼표로 나눈 결과다. "
                  "\"현 거주지\"(지역)와 \"성\"이 붙어 나온다. 이런 이름은 KOS-97 분류명 정규화에서 나눈다.",
                  "- 표 이름의 \"읍면동(연도 끝자리 0, 5), 시군구(그 외 연도)\"는 `region_level`과 `available_years`로는 "
                  "\"읍면동은 2020·2025년만\"을 나타내지 못한다(6절).", ""]

    lines += ["## 5. 평가 L0 evaluator와의 대응 (T2-28 `evals/evaluators/metadata.py`)", "",
              "| 이 스키마 (`TableFields`) | evaluator `fields()` | 차이 |", "| --- | --- | --- |",
              "| `available_years` | `years` | 이름만 다름 |",
              "| `regions = None` | `regions = \"none\"` | 전국 표 표기 |",
              "| `cycle` | `cycle` | 같은 규칙 (`record.cycle or record.prd_se`) |",
              "| `region_level`, `measure`, `dimensions` | 같은 이름 | 같은 함수 |",
              "| `measure_unit` | 없음 | 새 필드 |", "",
              "## 6. 알려진 한계", "",
              "- 지역 단위별 수록 연도가 다른 표(예: 읍면동은 연도 끝자리 0·5만, 그 외 해는 시군구까지)는 "
              f"표 단위 `available_years` 하나로 구분하지 못한다. 표 이름에 드러난 표는 "
              f"{sum('끝자리' in r.info.table_name for r in records)}개다. Parquet 값에서 단위별 연도를 읽는 일은 A7 또는 별도 이슈로 넘긴다.",
              "- `dimensions`는 표가 쓴 이름 그대로다(시도별·행정구역별(시도) 등). 표준 이름으로 묶는 것은 KOS-97이다.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "data" / "tables.jsonl")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.write_text(report(read_records(args.tables)), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
