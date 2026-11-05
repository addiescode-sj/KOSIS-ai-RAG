"""Rule checks on generated questions (T2-39, plans/T2-39.md Idea 3): the same rules for every table.

A question is dropped when it states something the table does not hold or copies the table:
    year       a year (19xx / 20xx, or a decade) the table has no value for (fields.available_years)
    sido       a 시도 name the table's regions do not include (filters.table_regions; tables without 시도 rows
               hold none); a bare 광주 also names 경기도 광주시, so it passes on a table that holds 경기도
    number     a numeric value that is not a year or an age (decimals, thousands, 4 digits that are not a 19xx / 20xx
               year, 5+ digits, a count / money / % unit)
    name_copy  the table name with almost no change (difflib ratio >= NAME_COPY)
    deictic    "이 표" and the like: the question does not say what it is about
    duplicate  almost the same as an earlier question of the same table (difflib ratio >= DUPLICATE)
Pure functions: no model, no I/O; check_positions gives the kept questions' places in the reply (their ids).
Thresholds are fixed constants, never tuned on golden questions.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from kosis_rag.fields import TableFields
from kosis_rag.hq.schemas import HqQuestion
from kosis_rag.schemas import TableRecord
from kosis_rag.text import norm_name
from kosis_rag.vocab import SIDO_LABELS

NAME_COPY = 0.9
DUPLICATE = 0.95
YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)(?!\s*년대)")
DECADE = re.compile(r"(?<!\d)(19\d|20\d)0\s*년대")
NUMBER = re.compile(r"(?<![\d.,])\d+(?:,\d{3}(?!\d))*(?:\.\d+)?")  # "3,000", "12.5", "2023"; not "2015, 2020" as one
# a number followed by one of these is a value ("3만 명", "12.5%", "500가구"); 세·살·대 (age), 월·분기, 인 (household
# size: "1인가구"), 개월, 년 are not values
VALUE_UNIT = re.compile(r"\s*(?:만|천|억|조|명|%|퍼센트|프로|가구|건|원|호|쌍|세대|배|채|㎡)")
SIDO_NAME = re.compile(
    r"(?<![가-힣])(" + "|".join(sorted(map(re.escape, SIDO_LABELS), key=len, reverse=True)) + r")"
    r"(?=$|[^가-힣]|시|도|특별|광역|은|는|이|가|의|에|과|와|도|별|로|을|를|만|및|지역|내|권)")
HOMONYM = {"광주": "경기도"}  # a bare short name that is also a 시군구 of this 시도 (경기도 광주시)
DEICTIC = re.compile(r"(?<![가-힣])(?:이|위|해당|본|아래)\s*(?:통계)?\s*표(?=$|[^가-힣]|에|의|는|를|로|가|와|과|만)")
_SPACE = re.compile(r"[\s·,.?!~\-()]+")


def _plain(text: str) -> str:
    return _SPACE.sub("", text).lower()


def ratio(a: str, b: str) -> float:
    """difflib similarity of two texts with spaces and punctuation removed (0..1).

    Side effects: none. Node: rule_check.
    """
    return SequenceMatcher(None, _plain(a), _plain(b)).ratio()


def bad_year(text: str, fields: TableFields) -> str | None:
    """Reason when the text names a year (or decade) the table has no value for; None when it names none or the
    table's years are unknown (an empty field means the source does not say).
    Side effects: none. Node: rule_check.
    """
    years = set(fields.available_years)
    if not years:
        return None
    for y in YEAR.findall(text):
        if int(y) not in years:
            return f"year: {y} not in {min(years)}~{max(years)}"
    for d in DECADE.findall(text):
        start = int(d) * 10
        if not years & set(range(start, start + 10)):
            return f"year: {start}년대 not in {min(years)}~{max(years)}"
    return None


def bad_sido(text: str, fields: TableFields) -> str | None:
    """Reason when the text names a 시도 the table does not hold (all of them, for a table without 시도 rows).

    Side effects: none. Node: rule_check.
    """
    held = set(fields.regions or [])
    named = set()
    for m in SIDO_NAME.findall(text):
        sido = SIDO_LABELS[m]
        if sido not in held and HOMONYM.get(m) in held:
            continue  # "광주" on a 경기도 table may be 경기도 광주시
        named.add(sido)
    missing = sorted(named - held)
    return f"sido: {', '.join(missing)} not in the table" if missing else None


def bad_number(text: str) -> str | None:
    """Reason when the text holds a numeric value other than a year or an age.

    Side effects: none. Node: rule_check.
    """
    for m in NUMBER.finditer(text):
        s = m.group(0)
        if ("." in s or "," in s or len(s) >= 5 or (len(s) == 4 and not YEAR.fullmatch(s))
                or VALUE_UNIT.match(text, m.end())):
            if YEAR.fullmatch(s) and not VALUE_UNIT.match(text, m.end()):
                continue  # a year in a list ("2015, 2020") is not a value
            return f"number: {s}"
    return None


def name_copy(text: str, record: TableRecord) -> str | None:
    """Reason when the text is the table name (as listed, or n2-normalized) with almost no change.

    Side effects: none. Node: rule_check.
    """
    names = [record.info.table_name, norm_name(record.info.table_name)[0]]
    best = max(ratio(text, n) for n in names)
    return f"name_copy: {best:.2f}" if best >= NAME_COPY else None


def deictic(text: str) -> str | None:
    """Reason when the text points at "this table" instead of saying what it asks for.

    Side effects: none. Node: rule_check.
    """
    m = DEICTIC.search(text)
    return f"deictic: {m.group(0)}" if m else None


def check_positions(questions: list[HqQuestion], record: TableRecord, fields: TableFields
                    ) -> tuple[list[int], dict[str, str]]:
    """Apply every rule to one table's questions, in reply order.

    In: questions = the validated reply; record = the table; fields = fields.table_fields(record).
    Out: (places in the reply of the kept questions, ascending; dropped question text -> the first rule it broke).
        A question that repeats an earlier kept one of this table (ratio >= DUPLICATE) is a duplicate.
    Side effects: none. Node: rule_check.
    """
    kept_at: list[int] = []
    dropped: dict[str, str] = {}
    for i, q in enumerate(questions):
        text = q.text.strip()
        reason = (bad_year(text, fields) or bad_sido(text, fields) or bad_number(text) or name_copy(text, record)
                  or deictic(text))
        if reason is None:
            dup = next((questions[j].text for j in kept_at if ratio(text, questions[j].text) >= DUPLICATE), None)
            reason = f"duplicate: {dup}" if dup is not None else None
        if reason is None:
            kept_at.append(i)
        else:
            dropped.setdefault(text, reason)
    return kept_at, dropped


def check_questions(questions: list[HqQuestion], record: TableRecord, fields: TableFields
                    ) -> tuple[list[HqQuestion], dict[str, str]]:
    """check_positions with the kept questions themselves: (kept questions in reply order, dropped text -> reason).

    Side effects: none. Node: rule_check (through check_positions).
    """
    kept_at, dropped = check_positions(questions, record, fields)
    return [questions[i] for i in kept_at], dropped
