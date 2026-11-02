"""Table text for the fine-tuned search models (T2-06, T2-07): one place for training (min `lab/`) and serving.

A model only works as measured when it sees the text it was trained on, so every format is named, the name is stored
with the Chroma collection (doc_format) and next to the weights, and a rule change means a new name and new training.
Every rule is a text pattern applied the same way to all tables (no table-specific fixes).

- "n2-L60": surface rules from the lab (min `lab/normalize_text.py`, KOS-88): the table name without its region tail
  and trailing brackets, "성/연령별" spelled out, "15세이상" -> "15세 이상", item units dropped, menu path tails, and
  label lists longer than 60 summarized. The first fine-tunings (plans 29, 43) used it.
- "n3-L60" (default): n2 plus what the master rules know (T2-27 roles, T2-30 vocab, T2-25 region summary): what the
  table counts ("세는 대상"), its breakdowns as standard keys ("분류"), its units, and region classes as level counts
  ("전국·시도(17)·시군구(250)") instead of place names. Place names stay in the BM25 text.

    build_doc(record, fmt) -> {"key", "embed_text" (embedding / re-ranker passage), "bm25_text" (full text), "fields"}
    norm_query(question) -> the question with the same surface rules (both formats)
"""

from __future__ import annotations

import re
from typing import Any

from kosis_rag.schemas import TableRecord

MAX_LABELS = 60  # longer label lists are summarized in the embedding text
FORMATS = ("n3-L60", "n2-L60")
DOC_FORMAT = FORMATS[0]  # default for new collections and training

GEO = r"(전국|시도|시군구|읍면동|동읍면|행정구역)"
GEO_TAIL = re.compile(r"\s*(?:-|–|/)\s*" + GEO + r"\s*$")
NUM_UNIT = re.compile(r"(\d+)\s*(세|년|명|인|개월|시간|분|㎡)\s*(이상|이하|미만|초과|전|후)")
YEAR_RANGE = re.compile(r"^\s*\d{4}\s*년?\s*([~∼-]\s*\d{4}\s*년?)?\s*$")
TRAILING_PAREN = re.compile(r"\s*\(([^()]*)\)\s*$")
TARGET_WORD = re.compile(r"\d+\s*세|\d+\+|가구|인구|내국인|외국인|주거전용|영업겸용")  # who is counted
BASIS = re.compile(r"\d{4}\s*년?\s*~|~\s*\d{4}|단위|기준|총액|순위|응답")  # how it is counted
UNIT_SUFFIX = re.compile(r"\s*\((명|가구|호|%|세|개|건|원|천원|백만원|㎡|년|개월|회|일|시간|분|천명|천가구)\)\s*$")
CYCLE = {"Y": "연", "H": "반기", "Q": "분기", "M": "월", "D": "일", "F": "다년(5년 등)", "IR": "부정기"}


def _unify(s: str) -> str:
    """Trim, single spaces, one form for wave dash and middle dot, "15세이상" -> "15세 이상"."""
    t = re.sub(r"\s+", " ", s.replace("　", " ")).strip().replace("∼", "~").replace("∙", "·")
    return NUM_UNIT.sub(r"\1\2 \3", t)


def norm_query(question: str) -> str:
    """Same surface rules on the question side, so "15세이상" in a query meets "15세 이상" in the index."""
    return _unify(question)


def norm_name(name: str) -> tuple[str, dict[str, str]]:
    """Table name -> (normalized name, {지역 수준, 대상, 기간, 기준}) taken off its tail."""
    s, fields = _unify(name), {}
    m = GEO_TAIL.search(s)
    if m:  # "… - 시군구": the region level is a field, not part of the name
        fields["지역 수준"] = "읍면동" if m.group(1) == "동읍면" else m.group(1)
        s = s[: m.start()]
    s = NUM_UNIT.sub(r"\1\2 \3", s)
    if s.count("(") > s.count(")") and re.search(r"\([^()]*$", s):  # source name lost its closing paren
        s += ")"
    target, period, basis = [], [], []
    while m := TRAILING_PAREN.search(s):  # only trailing parens; parens glued to a word stay with it
        inner = NUM_UNIT.sub(r"\1\2 \3", m.group(1).strip())
        if YEAR_RANGE.match(inner):
            period.insert(0, inner)
        elif all(TARGET_WORD.search(p) for p in inner.split(",")):  # "15세 이상", "20세 이상, 일반가구"
            target.insert(0, inner)
        elif BASIS.search(inner):  # "발생기준", "1년 단위", "2023~"
            basis.insert(0, inner)
        else:  # "그 외 연도", "출생,사망,혼인,이혼": part of the name, and nothing left of it peels off
            break
        s = s[: m.start()]
    if period:
        fields["기간"] = ", ".join(period)
    if basis:
        fields["기준"] = ", ".join(basis)
    if target:
        fields["대상"] = ", ".join(target)
    if "/" in s:  # "성/연령별 인구" -> "성별 연령별 인구"
        parts = [p.strip() for p in s.split("/") if p.strip()]
        last = parts[-1]
        m2 = re.match(r"^(.*?별)\s*(\S.*)$", last) if "별" in last else None  # "연령별 인구" or glued "유형별인구"
        if m2:
            head, subject = m2.group(1), m2.group(2)
        else:  # last part names the subject only ("5년 전 거주지 이동 인구"): keep it, no 별 added
            head, subject = None, last
        if any(" 및 " in d for d in parts[:-1]):  # "A 및 B보유/이용현황별": the slash joins two words of one category
            s = "·".join(parts)
        else:
            dims = ["성별" if d == "성" else d for d in [*parts[:-1], *([head] if head else [])]]
            s = " ".join(d if d.endswith("별") else d + "별" for d in dims) + (f" {subject}" if subject else "")
    return s.strip(), fields


def norm_label(label: str) -> str:
    return _unify(label)


def norm_item(item: str) -> str:
    s = UNIT_SUFFIX.sub("", _unify(item))  # the unit is already in the record's unit
    return s.replace("_", " ").strip()  # "내국인_남자" -> "내국인 남자"


def _menu_tails(rec: dict[str, Any]) -> list[str]:
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


def _dimension_names(dims: list[str]) -> list[str]:
    """Breakdowns as standard keys (vocab.py) where known, else as the table names them."""
    from kosis_rag.vocab import dimension_keys

    out: list[str] = []
    for d in dims:
        out += dimension_keys(d) or [d]
    return list(dict.fromkeys(x for x in out if x))


def _units(rec: TableRecord) -> list[str]:
    """Known units of the table: its own unit, then the units its item names end with ("혼인건수 (건)")."""
    from kosis_rag.vocab import unit_key

    found = [rec.unit] + [m.group(1) for i in rec.items if (m := TRAILING_PAREN.search(i))]
    return list(dict.fromkeys(u for u in map(unit_key, found) if u))


def build_doc(record: TableRecord | dict[str, Any], fmt: str = DOC_FORMAT, max_items: int = 30) -> dict[str, Any]:
    """One table -> embedding / re-ranker passage, full BM25 text and the fields peeled off its name."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown text format {fmt!r}; known: {', '.join(FORMATS)}")
    if not isinstance(record, TableRecord):
        record = TableRecord.model_validate(record)
    rec = record.model_dump()
    n3 = fmt.startswith("n3")
    max_labels = int(fmt.rsplit("-L", 1)[1])
    info = rec["info"]
    name, fields = norm_name(info["table_name"])
    items = list(dict.fromkeys(i for i in map(norm_item, rec.get("items", [])) if i))
    classes = [(_unify(c["name"]), list(dict.fromkeys(norm_label(x) for x in c["labels"] if x.strip())), c.get("levels"))
               for c in rec.get("classes", [])]
    cycle = "·".join(CYCLE.get(c, c) for c in rec.get("cycle", []))
    head = [f"통계표: {name}"]
    roles: list[str] = []
    if n3:
        from kosis_rag.roles import table_roles

        measure, dims = table_roles(record)
        if measure:
            roles.append(f"세는 대상: {_unify(measure)}")
        if names := _dimension_names(dims):
            roles.append("분류: " + "·".join(names))
    head += roles
    head.append(f"조사: {info['survey_name']} ({info['organization']})")
    if rec.get("topic"):
        head.append(f"주제: {rec['topic']}")
    head += [f"분류 경로: {t}" for t in _menu_tails(rec)]
    if rec.get("start") is not None:
        span = str(rec["start"]) if rec["start"] == rec["end"] else f"{rec['start']}~{rec['end']}"
        head.append(f"수록기간: {span}" + (f", 주기 {cycle}" if cycle else ""))
    head += [f"{k}: {v}" for k, v in fields.items()]

    embed = list(head)
    if items:
        more = f" 외 {len(items) - max_items}개" if len(items) > max_items else ""
        embed.append("항목: " + ", ".join(items[:max_items]) + more)
    if n3 and (units := _units(record)):
        embed.append("단위: " + ", ".join(units))
    for cname, labels, levels in classes:
        if n3 and levels:  # region class: which levels it covers, not hundreds of place names
            from kosis_rag.index import region_summary

            embed.append(f"{cname}: 지역 단위 {region_summary(levels)}")
        elif max_labels and len(labels) > max_labels:  # long lists (mostly 읍면동) would drown the topic
            embed.append(f"{cname}: {len(labels)}개 (예: {', '.join(labels[:5])})")
        else:
            embed.append(f"{cname}: {', '.join(labels)}")
    bm25 = [*head, info["table_name"], rec.get("table_name_en") or "", *items,
            *(f"{c} {' '.join(ls)}" for c, ls, _ in classes)]
    return {
        "key": info["key"],
        "embed_text": "\n".join(embed),
        "bm25_text": " ".join(p for p in bm25 if p),
        "fields": {**fields, "name": name, "format": fmt},
    }


def main(argv: list[str] | None = None) -> int:
    """Length per format (characters, and tokens with --tokens) and before/after samples for a few tables."""
    import argparse
    import random
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Table text formats: lengths and samples (kosis_rag.text)")
    parser.add_argument("--tables", type=Path, default=repo / "data" / "tables.jsonl")
    parser.add_argument("--tokens", action="store_true", help="also count bge-m3 tokens (downloads the tokenizer)")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--out", type=Path, default=repo / "data" / "text_formats_samples.md")  # data/ is gitignored
    args = parser.parse_args(argv)

    with args.tables.open(encoding="utf-8") as f:
        records = [TableRecord.model_validate_json(line) for line in f if line.strip()]
    tok = None
    if args.tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    docs = {fmt: [build_doc(r, fmt)["embed_text"] for r in records] for fmt in FORMATS}

    def pct(xs: list[int], q: float) -> int:
        return sorted(xs)[min(len(xs) - 1, int(len(xs) * q))]

    print(f"{len(records)} tables")
    for fmt, texts in docs.items():
        chars = [len(t) for t in texts]
        line = f"  {fmt}: chars median {pct(chars, .5)}, p90 {pct(chars, .9)}, max {max(chars)}"
        if tok is not None:
            n = [len(tok(t)["input_ids"]) for t in texts]
            line += (f" | tokens median {pct(n, .5)}, p90 {pct(n, .9)}, max {max(n)}, "
                     f"> 480 (re-ranker pair 512 minus a question) {sum(x > 480 for x in n)}, > 1024 {sum(x > 1024 for x in n)}")
        print(line)
    random.seed(0)
    picks = random.sample(range(len(records)), min(args.samples, len(records)))
    with args.out.open("w", encoding="utf-8") as f:
        for i in picks:
            f.write(f"## {records[i].info.key}\n원문 표 이름: {records[i].info.table_name}\n\n")
            for fmt in FORMATS:
                f.write(f"### {fmt}\n```\n{docs[fmt][i]}\n```\n\n")
    print(f"samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
