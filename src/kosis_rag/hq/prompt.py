"""The fixed question-generation prompt (T2-39, plans/T2-39.md Idea 2): one table in, six questions out.

The same template is used for every table; only the table's n2 text and its field summary are filled in. No golden
question or answer is used anywhere in it. Changing the wording, the version or the generation settings changes
`prompt_hash()`, so replies cached under the old prompt are never reused (data/hq/cache/<prompt_hash8>/<model>/;
the table's own user message is keyed in the file name, hq.graph.cache_path).
"""

from __future__ import annotations

import hashlib

from kosis_rag.fields import TableFields
from kosis_rag.schemas import TableRecord
from kosis_rag.text import CYCLE

PROMPT_VERSION = "hq-v1"
TEMPERATURE = 0.7  # some variety between the questions of one table; the seed keeps a rerun close
SEED = 0
MAX_TOKENS = 1024
COUNTS = {"concept": 2, "condition": 2, "distinct": 1, "keyword": 1}  # D5: six questions
MAX_LISTED_YEARS = 40  # longer gappy year lists are summarized as a range with a count

SYSTEM = (
    "당신은 KOSIS 통계표 검색 색인을 만드는 도우미입니다. 통계표 설명을 읽고, 사람이 이 통계표를 찾을 때 "
    "검색창이나 챗봇에 입력할 법한 질문을 만드세요. 질문에 답하지 마세요.\n\n"
    "# 만들 것\n"
    "- description: 이 통계표가 무엇을 무엇별로 보여 주는지 한 문장.\n"
    f"- questions: 질문 {sum(COUNTS.values())}개. 유형별 개수를 지키세요.\n"
    f"  1. concept {COUNTS['concept']}개: 일반인 말투. 통계표 이름과 항목 이름의 단어를 되도록 피하고 뜻으로 말하세요"
    "(예: '1인가구' 대신 '혼자 사는 사람').\n"
    f"  2. condition {COUNTS['condition']}개: 이 통계표에 실제로 있는 연도·지역·성별·연령 같은 조건을 넣으세요.\n"
    f"  3. distinct {COUNTS['distinct']}개: 이 통계표만의 주기(연간·월간·5년 등), 지역 수준(시도·시군구·읍면동), "
    "수록 기간이 드러나는 질문.\n"
    f"  4. keyword {COUNTS['keyword']}개: 검색창에 치듯 2~4단어.\n\n"
    "# 지킬 것\n"
    "- 연도는 '수록 연도'에 있는 것만 쓰세요. 시도 이름은 '시도'에 있는 것만 쓰세요.\n"
    "- 수치 값(인구 수, 가구 수, 비율, 금액 등)을 질문에 넣지 마세요. 연도와 나이는 괜찮습니다.\n"
    "- 통계표 이름을 그대로 베끼지 마세요.\n"
    "- '이 표', '위 통계표', '해당 표' 같은 지시어를 쓰지 마세요. 질문만 읽어도 무엇을 찾는지 알 수 있어야 합니다.\n"
    "- 모든 질문은 한국어로, 서로 다른 내용으로 쓰세요.\n\n"
    "# 출력 형식\n"
    "설명이나 Markdown 없이 JSON 객체 하나만 출력하세요.\n"
    '{"description": "...", "questions": [{"type": "concept", "text": "..."}, '
    '{"type": "condition", "text": "..."}, {"type": "distinct", "text": "..."}, {"type": "keyword", "text": "..."}]}\n'
    "type은 concept, condition, distinct, keyword 중 하나입니다."
)
USER = "통계표 설명:\n{doc}\n\n필드 요약:\n{summary}"
REASK = ("앞의 출력은 형식이 맞지 않습니다: {error}\n"
         "설명 없이 형식에 맞는 JSON 객체 하나만 다시 출력하세요.")


def prompt_hash() -> str:
    """sha256 of the fixed template, its version and the generation settings (not of any table's content).

    In: nothing. Out: 64 hex characters; its first 8 name the cache folder and go into the build id.
    Side effects: none. Node: prompt.
    """
    parts = [PROMPT_VERSION, SYSTEM, USER, REASK, f"t={TEMPERATURE}", f"seed={SEED}", f"max={MAX_TOKENS}"]
    return hashlib.sha256("\n\x1e".join(parts).encode("utf-8")).hexdigest()


def years_text(years: list[int]) -> str:
    """'2015~2025 (매년)', '2000, 2005, 2010, 2015, 2020' or '1925~2020 중 45개 연도'; '알 수 없음' when empty.

    Side effects: none. Node: prompt.
    """
    if not years:
        return "알 수 없음"
    if len(years) == 1:
        return str(years[0])
    if years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}~{years[-1]} (매년)"
    if len(years) <= MAX_LISTED_YEARS:
        return ", ".join(map(str, years))
    return f"{years[0]}~{years[-1]} 중 {len(years)}개 연도"


def field_summary(record: TableRecord, fields: TableFields) -> str:
    """The facts the questions may use, one per line: measure, breakdowns, years, cycle, region level, 시도, unit.

    In: record = the table; fields = fields.table_fields(record). Out: Korean lines for the user message.
    Side effects: none. Node: prompt.
    """
    classes = fields.dimensions or [c.name for c in record.classes]
    cycle = "·".join(CYCLE.get(c, c) for c in fields.cycle) or "알 수 없음"
    if fields.regions is None:
        sido = "없음 (시도별 구분이 없는 표)"
    elif len(fields.regions) >= 17:
        sido = f"{len(fields.regions)}개 시도 전체"
    else:
        sido = ", ".join(fields.regions)
    return "\n".join([
        f"측정 대상: {fields.measure or record.info.table_name}",
        f"분류: {', '.join(classes) or '없음'}",
        f"수록 연도: {years_text(fields.available_years)}",
        f"주기: {cycle}",
        f"지역 수준: {'·'.join(['전국', *fields.region_level]) if fields.region_level else '전국 또는 구분 없음'}",
        f"시도: {sido}",
        f"단위: {fields.measure_unit or record.unit or '알 수 없음'}",
    ])


def build_messages(record: TableRecord, fields: TableFields, doc: str) -> list[dict[str, str]]:
    """[system, user] chat messages for one table.

    In: record, fields = the table and its fields; doc = its n2 embedding text. Out: two {"role", "content"} dicts.
    Side effects: none. Node: prompt.
    """
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(doc=doc, summary=field_summary(record, fields))}]
