"""KOS-88: turn one data/tables.jsonl record into search text with global rules only.

Every rule is a text pattern applied the same way to all 2,364 tables (no table-specific fixes).
build_doc(record) returns
    embed_text  short passage for the embedding model (long label lists summarized)
    bm25_text   full normalized text for BM25 (every label kept)
    fields      parsed facts (region level, target group, period) for metadata filters

    python lab/normalize_text.py                    # rule counts + 30 random before/after samples
    python lab/normalize_text.py --comma --pad      # with the optional rules R5 / L5
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables.jsonl"
OUT_DIR = ROOT / "data" / "lab"  # gitignored (/data/)

RULES_VERSION = "n2"  # bump when a rule changes; goes into collection names (KOS-90)

GEO = r"(전국|시도|시군구|읍면동|동읍면|행정구역)"
GEO_TAIL = re.compile(r"\s*(?:-|–|/)\s*" + GEO + r"\s*$")
NUM_UNIT = re.compile(r"(\d+)\s*(세|년|명|인|개월|시간|분|㎡)\s*(이상|이하|미만|초과|전|후)")
YEAR_RANGE = re.compile(r"^\s*\d{4}\s*년?\s*([~∼-]\s*\d{4}\s*년?)?\s*$")
TRAILING_PAREN = re.compile(r"\s*\(([^()]*)\)\s*$")
TARGET_WORD = re.compile(r"\d+\s*세|\d+\+|가구|인구|내국인|외국인|주거전용|영업겸용")  # who is counted
BASIS = re.compile(r"\d{4}\s*년?\s*~|~\s*\d{4}|단위|기준|총액|순위|응답")  # how it is counted
UNIT_SUFFIX = re.compile(r"\s*\((명|가구|호|%|세|개|건|원|천원|백만원|㎡|년|개월|회|일|시간|분|천명|천가구)\)\s*$")
COMMA_LIST = re.compile(r"^[^,]+(, [^,]+)* 및 [^ ]+별 ")
PADDED = re.compile(r"^[가-힣](\s{2,}[가-힣])+$")
SPACED_DONG = re.compile(r"([가-힣]) (\d+[동가리])$")
CYCLE = {"Y": "연", "H": "반기", "Q": "분기", "M": "월", "D": "일", "F": "다년(5년 등)", "IR": "부정기"}

hits: collections.Counter[str] = collections.Counter()


def _unify(s: str) -> str:
    """L1 + L2: trim, single spaces, one form for wave dash and middle dot."""
    t = re.sub(r"\s+", " ", s.replace("　", " ")).strip().replace("∼", "~").replace("∙", "·")
    return NUM_UNIT.sub(r"\1\2 \3", t)  # R1 for labels, items and queries too


def norm_query(question: str) -> str:
    """Same surface rules on the question side, so "15세이상" in a query meets "15세 이상" in the index."""
    return _unify(question)


def norm_name(name: str, *, comma: bool = False) -> tuple[str, dict]:
    """Table name -> (normalized name, {지역 수준, 대상, 기간})."""
    s, fields = _unify(name), {}
    m = GEO_TAIL.search(s)
    if m:  # R3
        fields["지역 수준"] = "읍면동" if m.group(1) == "동읍면" else m.group(1)
        s = s[: m.start()]
        hits["R3 지역 꼬리표"] += 1
    t = NUM_UNIT.sub(r"\1\2 \3", s)
    if t != s:  # R1
        hits["R1 숫자 띄어쓰기"] += 1
    s = t
    if s.count("(") > s.count(")") and re.search(r"\([^()]*$", s):  # source name lost its closing paren
        hits["R4c 닫는 괄호 보정"] += 1
        s += ")"
    target, period, basis = [], [], []
    while m := TRAILING_PAREN.search(s):  # R4: only trailing parens; parens glued to a word stay with it
        inner = NUM_UNIT.sub(r"\1\2 \3", m.group(1).strip())
        if YEAR_RANGE.match(inner):
            period.insert(0, inner)
        elif all(TARGET_WORD.search(p) for p in inner.split(",")):  # "15세 이상", "20세 이상, 일반가구"
            target.insert(0, inner)
        elif BASIS.search(inner):  # "발생기준", "1년 단위", "2023~"
            basis.insert(0, inner)
        else:  # "그 외 연도", "출생,사망,혼인,이혼", "읍면동/성/연령별": part of the name, and nothing left of it peels off
            break
        s = s[: m.start()]
    if target or period or basis:
        hits["R4 끝 괄호"] += 1
    if period:
        hits["R4b 기간 괄호"] += 1
        fields["기간"] = ", ".join(period)
    if basis:
        hits["R4d 기준 괄호"] += 1
        fields["기준"] = ", ".join(basis)
    if target:
        hits["R4e 대상 괄호"] += 1
        fields["대상"] = ", ".join(target)
    if "/" in s:  # R2
        hits["R2 슬래시 풀기"] += 1
        parts = [p.strip() for p in s.split("/") if p.strip()]
        last = parts[-1]
        m2 = re.match(r"^(.*?별)\s*(\S.*)$", last) if "별" in last else None  # "연령별 인구" or glued "유형별인구"
        if m2:
            head, subject = m2.group(1), m2.group(2)
        else:  # last part names the subject only ("5년 전 거주지 이동 인구"): keep it, no 별 added
            head, subject = None, last
        if any(" 및 " in d for d in parts[:-1]):  # "A 및 B보유/이용현황별": the slash joins two words of one category
            hits["R2 슬래시=가운뎃점"] += 1
            s = "·".join(parts)
        else:
            dims = ["성별" if d == "성" else d for d in [*parts[:-1], *([head] if head else [])]]
            s = " ".join(d if d.endswith("별") else d + "별" for d in dims) + (f" {subject}" if subject else "")
    elif comma and COMMA_LIST.match(s) and not re.match(r"^\S+의 ", s):  # R5 (optional)
        hits["R5 쉼표 나열"] += 1
        head, subject = re.match(r"^(.*?별)\s+(.+)$", s).groups()
        s = " ".join("성별" if d == "성" else d + "별" for d in re.split(r", | 및 ", head[:-1])) + f" {subject}"
    return s.strip(), fields


def norm_label(label: str, *, pad: bool = False) -> str:
    s = _unify(label)
    if pad and PADDED.match(label.strip()):  # L5 (optional): "남  면" -> "남면"
        s = re.sub(r"\s+", "", s)
    if pad:
        s = SPACED_DONG.sub(r"\1\2", s)  # "수유 1동" -> "수유1동"
    return s


def norm_item(item: str) -> str:
    s = UNIT_SUFFIX.sub("", _unify(item))  # L3: unit is already in record["unit"]
    return s.replace("_", " ").strip()  # L4: "내국인_남자" -> "내국인 남자"


def _menu_tails(rec: dict) -> list[str]:
    """Last two steps of every menu path: tells apart tables that share a name."""
    name = rec["info"]["table_name"].strip()
    tails = []
    for p in rec.get("menu_paths", []):
        steps = [x.strip() for x in p.split(" > ") if x.strip()]
        if steps and steps[-1] == name:  # the path usually ends with the table name itself
            steps = steps[:-1]
        if steps:
            tails.append(" > ".join(steps[-2:]))
    return list(dict.fromkeys(tails))


def build_doc(rec: dict, *, max_labels: int = 60, max_items: int = 30, comma: bool = False, pad: bool = False) -> dict:
    info = rec["info"]
    name, fields = norm_name(info["table_name"], comma=comma)
    items = list(dict.fromkeys(i for i in map(norm_item, rec.get("items", [])) if i))
    classes = [(_unify(c["name"]), list(dict.fromkeys(norm_label(l, pad=pad) for l in c["labels"] if l.strip())))
               for c in rec.get("classes", [])]
    cycle = "·".join(CYCLE.get(c, c) for c in rec.get("cycle", []))
    head = [f"통계표: {name}", f"조사: {info['survey_name']} ({info['organization']})"]
    if rec.get("topic"):
        head.append(f"주제: {rec['topic']}")
    head += [f"분류 경로: {t}" for t in _menu_tails(rec)]
    if rec.get("start") is not None:
        span = str(rec["start"]) if rec["start"] == rec["end"] else f"{rec['start']}~{rec['end']}"
        head.append(f"수록기간: {span}" + (f", 주기 {cycle}" if cycle else ""))
    head += [f"{k}: {v}" for k, v in fields.items()]

    embed = list(head)
    if items:
        embed.append("항목: " + ", ".join(items[:max_items]) + (f" 외 {len(items) - max_items}개" if len(items) > max_items else ""))
    for cname, labels in classes:
        if max_labels and len(labels) > max_labels:  # long lists (mostly 읍면동) would drown the topic; 0 = off
            # experimental: may go away when chunking (S2) or another text recipe replaces it
            hits["E 라벨 요약"] += 1
            embed.append(f"{cname}: {len(labels)}개 (예: {', '.join(labels[:5])})")
        else:
            embed.append(f"{cname}: {', '.join(labels)}")
    bm25 = [*head, info["table_name"], rec.get("table_name_en") or "", *items,
            *(f"{c} {' '.join(ls)}" for c, ls in classes)]
    return {
        "key": info["key"],
        "embed_text": "\n".join(embed),
        "bm25_text": " ".join(p for p in bm25 if p),
        "fields": {**fields, "name": name, "rules": RULES_VERSION},
    }


def load_records(path: Path = TABLES) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comma", action="store_true", help="optional rule R5")
    p.add_argument("--pad", action="store_true", help="optional rule L5")
    p.add_argument("--max-labels", type=int, default=60, help="summarize longer label lists; 0 = never")
    p.add_argument("--samples", type=int, default=30)
    args = p.parse_args()

    recs = load_records()
    docs = [build_doc(r, max_labels=args.max_labels, comma=args.comma, pad=args.pad) for r in recs]
    lens = sorted(len(d["embed_text"]) for d in docs)
    print(f"{len(docs)} tables, rules {RULES_VERSION}, comma={args.comma}, pad={args.pad}, max_labels={args.max_labels}")
    for rule, n in sorted(hits.items()):
        print(f"  {rule}: {n}")
    print(f"  embed_text chars: median {lens[len(lens) // 2]}, p90 {lens[int(len(lens) * 0.9)]}, max {lens[-1]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"normalize_samples_{RULES_VERSION}.md"
    random.seed(0)
    picks = random.sample(range(len(recs)), min(args.samples, len(recs)))
    with out.open("w", encoding="utf-8") as f:
        for i in picks:
            f.write(f"## {recs[i]['info']['key']}\n원문: {recs[i]['info']['table_name']}\n\n```\n{docs[i]['embed_text']}\n```\n\n")
    print(f"samples -> {out}")


if __name__ == "__main__":
    main()
