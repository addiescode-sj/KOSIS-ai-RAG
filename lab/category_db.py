"""Experiment 14: one search space per topic category, with the category's own words taken out.

Hypothesis (docs: phase2-lab/14-category-db-plan.md): topic words shared by many tables (출생, 혼인, 가구 ...)
make every vector in a space alike and bury what tells tables apart. Inside the 출생 space every document and
the question lose the 출생 words, so tables compete on what is left (region level, measure, period).

Categories are the topic names of the KOSIS menu (not taken from the golden questions). A table belongs to every
category whose words appear in its name, items or menu path; a table with none goes to 기타.

    python lab/category_db.py          # category sizes, overlap, and a before/after sample per category
"""

from __future__ import annotations

import collections
import re

# KOSIS menu topic names -> the words that mark them (13 문서). Order only matters for printing.
CATEGORIES: dict[str, str] = {
    "출생": r"출생|출산|자녀수",
    "사망": r"사망",
    "혼인": r"혼인|결혼|초혼|재혼",
    "이혼": r"이혼",
    "가구": r"가구|세대",
    "주거": r"주거|주택|거처|임차|전세|월세|사글세|점유형태|방수|난방",
    "경제활동": r"경제활동|취업|종사상|산업|직업",
    "통근통학": r"통근|통학",
    "이동": r"이동|전입|전출|거주지",
    "외국인": r"외국인|외국국적|귀화|체류|다문화",
    "출입국": r"출입국|입국|출국",
    "고령": r"고령|노인",
    "여성·아동": r"아동|보육|기혼여성|경력단절|육아",
    "활동제약": r"활동제약|장애",
    "교육": r"교육정도|학력|재학|학교",
    "추계": r"추계|장래",
    "사할린동포": r"사할린|영주귀국",
    "인구일반": r"인구밀도|총인구|주민등록|인구수|연앙인구",
}
OTHER = "기타"
PATTERNS = {c: re.compile(p) for c, p in CATEGORIES.items()}
WORD = re.compile(r"[가-힣A-Za-z]{2,}")


def categories_of(text: str) -> set[str]:
    return {c for c, p in PATTERNS.items() if p.search(text)}


def table_categories(rec: dict) -> set[str]:
    """Name + items + menu path (plan 14, detail 2). Empty -> {기타}."""
    text = " ".join([rec["info"]["table_name"], *rec.get("items", []), *rec.get("menu_paths", [])])
    return categories_of(text) or {OTHER}


def strip_words(text: str, category: str) -> str:
    """Delete the category's words as plain characters (plan 14, detail 1): 출생건수 -> 건수.
    Lists and lines left empty by the deletion are tidied away."""
    if category == OTHER:
        return text
    t = PATTERNS[category].sub("", text)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"(?:,\s*){2,}", ", ", t)
    lines = []
    for line in t.split("\n"):
        line = re.sub(r":\s*,\s*", ": ", line.strip())
        line = re.sub(r"[,\s]+$", "", line)
        if line and not line.endswith(":"):  # drop "항목:" left with nothing
            lines.append(line)
    return "\n".join(lines)


def strip_query(question: str, category: str) -> str:
    """Same deletion on the question; keep the original when too little is left ("출생 통계 알려줘")."""
    t = re.sub(r"\s+", " ", strip_words(question, category)).strip()
    return t if len(WORD.findall(t)) >= 2 else question


def route_rule(question: str) -> set[str]:
    """Rule router: the same word list on the question. Empty set = no category, search everything."""
    return categories_of(question)


def build_spaces(records: list[dict]) -> dict[str, list[int]]:
    """category -> indices of its tables (a table can sit in several)."""
    spaces: dict[str, list[int]] = collections.defaultdict(list)
    for i, rec in enumerate(records):
        for c in table_categories(rec):
            spaces[c].append(i)
    return dict(spaces)


def main() -> None:
    from normalize_text import build_doc, load_records

    records = load_records()
    spaces = build_spaces(records)
    member = collections.Counter(len(table_categories(r)) for r in records)
    print(f"{len(records)} tables, {len(spaces)} spaces, documents over all spaces {sum(map(len, spaces.values()))}")
    print("categories per table: " + ", ".join(f"{k}: {v}" for k, v in sorted(member.items())))
    for c in [*CATEGORIES, OTHER]:
        if c not in spaces:
            continue
        i = spaces[c][0]
        before = build_doc(records[i])["embed_text"].split("\n")
        after = strip_words("\n".join(before), c).split("\n")
        print(f"\n[{c}] {len(spaces[c])} tables  e.g. {records[i]['info']['key']}")
        print("  before: " + " | ".join(before[:1] + [l for l in before if l.startswith("항목")])[:220])
        print("  after : " + " | ".join(after[:1] + [l for l in after if l.startswith("항목")])[:220])


if __name__ == "__main__":
    main()
