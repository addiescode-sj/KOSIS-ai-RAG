"""T2-30 (KOS-96): profile the per-table fields over data/tables.jsonl and write docs/reports/t2-30-fields.md.

Only aggregates and catalog tables go in the report; no golden question or answer key (T2-02 rule).

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/field_profile.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from kosis_rag.adapters.files import load_sigungu
from kosis_rag.index import REPO, read_records
from kosis_rag.shared.fields import ITEM_UNITS, LAST_BRACKET, TableFields, item_unit, table_fields
from kosis_rag.shared.vocab import SIDO_LABELS, dimension_keys, unit_key

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


def _a2_section(records, fields: list[TableFields]) -> list[str]:
    """KOS-97: how much of each vocabulary the shared names (vocab.py) cover."""
    names = [d for f in fields for d in f.dimensions]
    unmapped = [d for d in names if not dimension_keys(d)]
    sigungu = load_sigungu()
    top = {"전국", "계", "총계", "합계", "동부", "읍부", "면부"}
    region_total = region_miss = 0
    old: Counter = Counter()
    for r in records:
        for c in r.classes:
            if "시군구" not in c.levels or c.levels.get("읍면동"):
                continue  # 시군구 classes only; 읍면동 names are not in the 시군구 map
            for label in c.labels:
                if label in top or label in SIDO_LABELS:
                    continue
                region_total += 1
                if label not in sigungu:
                    region_miss += 1
                    old[label] += 1
    units = Counter(unit_key(f.measure_unit) for f in fields)
    n = len(fields)
    lines = ["## 5. 정규화 사전과 미매핑 비율 (KOS-97)", "",
             "사전은 `src/kosis_rag/shared/vocab.py` 한 곳에 둔다. 시도 이름(`filters.SIDO`)·질문 지역어(`query.REGION_WORDS`)는 "
             "제자리에 두고 같은 객체를 다시 내보낸다. 시군구 → 시도는 `python -m kosis_rag.shared.vocab --data-dir <제공 데이터>`로 "
             "만든 `data/sigungu.json`(git 제외)을 쓴다.", "",
             "| 사전 | 대상 | 미매핑 | 비고 |", "| --- | --- | ---: | --- |",
             f"| 분류 표준 이름 | 분류 이름 등장 {len(names):,}회 | {len(unmapped) / len(names):.1%} ({len(unmapped):,}회) | "
             f"고유 이름 {len(set(names)):,}개 중 {len(set(unmapped)):,}개 미매핑(긴 꼬리), 표준 이름 "
             f"{len({k for d in names for k in dimension_keys(d)})}개 |",
             f"| 시군구 → 시도 | 시군구 분류 라벨 {region_total:,}개 | "
             f"{region_miss / max(region_total, 1):.1%} ({region_miss:,}개) | {len(sigungu)}개 시군구. 미매핑은 대부분 옛 이름 |",
             f"| 단위 표기 | 표 {n:,}개 | {units[None] / n:.1%} ({units[None]:,}개) | 원본에 단위가 없는 표. 배율 변환은 하지 않음 |", "",
             "미매핑 분류 상위: " + ", ".join(f"{v} {c}" for v, c in Counter(unmapped).most_common(10)),
             "", "미매핑 시군구 상위(통합·개편 전 이름): " + ", ".join(f"{v} {c}" for v, c in old.most_common(8)), "",
             "- 평가 L0의 `dimension_unmapped`는 \"질문 분류어 13개 중 하나로 걸러질 수 있는가\"를 재는 다른 정의다. "
             "위 수치는 \"표준 이름 표에 붙었는가\"이다.",
             "- 질문 쪽 별칭: \"1인 가구\"~\"6인 가구\", \"7인 이상 가구\" → 가구원수 분류의 라벨(`vocab.LABEL_ALIASES`).", ""]
    return lines


def _a7_section(records, fields: list[TableFields]) -> list[str]:
    """KOS-109: breakdowns found outside the classes and the measure for names without the usual shape."""
    implied = Counter(k for f in fields for k in f.implied_keys)
    fixed = Counter(x for f in fields for x in f.fixed)
    lines = ["## 6. 분류 밖에 있는 조건과 세는 대상 보강 (KOS-109, 평가 개선 문서 D1~D7)", "",
             "분류 이름이 아닌 곳(세는 대상·항목 이름·라벨)에 있는 조건은 `implied_keys`에, 한 성만 대상인 표는 `fixed`에 "
             "따로 둔다. `dimension_keys`(원래 분류)와 섞지 않아 규칙이 틀려도 원래 분류가 오염되지 않고, 필터가 이 필드를 "
             "어떻게 쓸지는 필터 담당이 정한다.", "",
             "| 규칙 | 키 | 표 수 |", "| --- | --- | ---: |",
             f"| D1 세는 대상에 \"N인가구\" | 가구원수 | {implied['가구원수']:,} |",
             f"| D2 항목에 남·여가 둘 다 / 혼인상태 값 2개 이상 / \"…년생\" 분류 | 성 · 혼인상태 · 연령 | "
             f"{implied['성'] - sum(fixed.values()):,} · {implied['혼인상태']:,} · {implied['연령']:,} |",
             f"| D3 이름·세는 대상이 한 성만 | 성 + `fixed` | 여성 {fixed['성=여성']:,} · 남성 {fixed['성=남성']:,} |",
             f"| D5 단위 정보 없는 지역 분류(\"도별\"·\"지역별\"·\"대도시별\")의 라벨이 시도 이름 | 시도 | {implied['시도']:,} |",
             f"| 합계 (표 기준, 겹침 제외) | | {sum(bool(f.implied_keys) for f in fields):,} |", "",
             "- 단어 목록은 2,364표 항목 이름의 단어 빈도에서 골랐다: 남자·여자 각 462표, 남·여 62·61, 남편·아내 24·25, "
             "미혼 60, 사별 66, 이혼 74, 유배우 30, 배우자있음 17.",
             "- D4 `measure_keys`: 표기만 하나로(공백 제거, \"·\" 목록은 각 부분도). 고령자 → 인구 같은 뜻 넓히기는 하지 않는다.",
             "- D6: 이름이 \"A별 X\" 꼴이 아닌 표도 이름이 나열하는 것(\"출생아수, 합계출산율, 자연증가 등\"), "
             "현황·추이·분포·구조 앞 말, 항목이 하나뿐이면 그 항목을 세는 대상으로 읽는다. \"비율\"처럼 방법만 말하는 항목은 이름이 "
             "그 말로 끝날 때만 이름을 쓴다(설문 문항 표는 비워 둠). 이미 있던 값은 바뀌지 않는다.",
             "- D7: 표준 이름이 없는 분류는 정규화한 이름(공백·\"별\"·괄호 수식어 제거, `vocab.normalize_dimension`)을 그대로 키로 "
             "쓴다(예: \"전입사유별\" → `전입사유`). 그래서 `dimension_keys` 채움률은 성과가 아니고, 표준 이름 매핑률은 5절의 "
             "미매핑 비율로 따로 본다.", ""]
    return lines


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
              "이 가운데 시도 이름이 출생지·이전 거주지·통근통학지 같은 분류에만 있는 표는 1개뿐이라 `regions` 규칙은 그대로 둔다(KOS-97).",
              "- `measure`는 표 이름이 \"A/B/C별 X\" 꼴이 아닌 표에서 비워 둔다(T2-27). KOS-97에서 묶음형 표"
              "(\"인구동태건수 및 동태율(출생,사망,혼인,이혼)\" → `출생·사망·혼인·이혼`), \"…별인구\"처럼 붙어 쓴 이름, "
              "\"/ 전국\" 꼬리를 읽도록 보강해 2,071 → 2,117표가 됐다. 이미 채워져 있던 값은 바뀌지 않았다.", "",
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
                  "\"읍면동은 2020·2025년만\"을 나타내지 못한다(8절).", ""]

    lines += _a2_section(records, fields)
    lines += _a7_section(records, fields)
    lines += ["## 7. 평가 L0 evaluator와의 대응 (T2-28 `evals/evaluators/metadata.py`)", "",
              "evaluator `fields()`가 `table_fields()`를 그대로 부른다. 필드 이름·값이 같고, "
              "전국 표의 `regions = None`은 채움으로 센다. `measure_unit`·`dimension_keys`도 채움률을 잰다. "
              "KOS-109부터 `dimension_keys`에는 표준 이름이 없는 분류의 정규화한 이름도 들어가므로(6절 D7), "
              "표준 이름 매핑률은 5절의 미매핑 비율로 본다. 새 필드 `implied_keys`·`fixed`·`measure_keys`는 따로 있다.", "",
              "## 8. 알려진 한계", "",
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
