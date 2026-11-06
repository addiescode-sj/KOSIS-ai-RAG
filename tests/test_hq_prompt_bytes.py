"""T2-41 S6a: the hq prompt moved to src/kosis_rag_build/hq/prompts/hq-v1.md is the bytes of the old string constants.

OLD_SYSTEM / OLD_USER / OLD_REASK are the old kosis_rag.hq.prompt expressions copied verbatim (with the old COUNTS),
and OLD_SHA256 the sha256 of the values that module produced, taken before it was deleted. prompt_hash() names the
reply cache folder and goes into every build id: its first 8 characters must stay "cb0b3c5b" (the PC builds
data/hq/hq-cb0b3c5b-*), or a replay refuses to start and a rebuild calls the API for every table.
Kept for the moving step (as tests/test_prompts_bytes.py for S3 / S5); the hash check stays useful after it.
"""

import hashlib
from importlib import resources

from kosis_rag_build.hq import constants, prompt

COUNTS = {"concept": 2, "condition": 2, "distinct": 1, "keyword": 1}
OLD_SHA256 = {
    "system": "fde21f081b8f0688fec862b63bdfea4619873a5d88f6bedca09b22637567e2b0",
    "user": "100477ac2603cc3f74925ea0a0e6701ff5455b7c4f1bb7486763831afa74efb2",
    "reask": "a4351473470fd345214a2bf08ce78c3226160d6fca24641d8d2f3280b98002f9",
}
OLD_PROMPT_HASH = "cb0b3c5b7ed78c0b97358e0f2c6245392b474580442204568e9c5783e3a0f68c"

OLD_SYSTEM = (
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
OLD_USER = "통계표 설명:\n{doc}\n\n필드 요약:\n{summary}"
OLD_REASK = ("앞의 출력은 형식이 맞지 않습니다: {error}\n"
             "설명 없이 형식에 맞는 JSON 객체 하나만 다시 출력하세요.")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_old_expressions_give_the_recorded_values():
    assert {"system": _sha(OLD_SYSTEM), "user": _sha(OLD_USER), "reask": _sha(OLD_REASK)} == OLD_SHA256


def test_the_prompt_file_reads_back_as_the_old_constants():
    p = prompt.prompt_texts()
    assert (p.system, p.user, p.reask) == (OLD_SYSTEM, OLD_USER, OLD_REASK)
    assert {"system": _sha(p.system), "user": _sha(p.user), "reask": _sha(p.reask)} == OLD_SHA256


def test_prompt_hash_is_unchanged():
    assert prompt.prompt_hash() == OLD_PROMPT_HASH
    assert prompt.prompt_hash()[:8] == "cb0b3c5b"  # the PC build ids hq-cb0b3c5b-...


def test_the_settings_in_the_hash_are_unchanged():
    assert (constants.PROMPT_VERSION, constants.TEMPERATURE, constants.SEED, constants.MAX_TOKENS) == ("hq-v1", 0.7, 0, 1024)
    assert constants.COUNTS == COUNTS


def test_the_file_is_package_data_with_the_separators_once():
    raw = (resources.files("kosis_rag_build.hq") / "prompts" / constants.PROMPT_FILE).read_bytes()
    text = raw.decode("utf-8")
    assert b"\r" not in raw and text.count(prompt.USER_SEPARATOR) == 1 and text.count(prompt.REASK_SEPARATOR) == 1
    assert text == OLD_SYSTEM + prompt.USER_SEPARATOR + OLD_USER + prompt.REASK_SEPARATOR + OLD_REASK


def test_the_messages_are_built_from_the_file():
    from kosis_rag.schemas.catalog import TableRecord
    from kosis_rag.schemas.common import TableInfo
    from kosis_rag.shared.fields import table_fields

    info = TableInfo(key="101_DT_X", org_id="101", tbl_id="DT_X", survey_name="s", organization="o", table_name="t",
                     period="년", source="api", coverage="A", rows=1, numeric_rows=1, symbol_rows=0,
                     parquet="values/x.parquet", url="u", menu_path="m")
    record = TableRecord(info=info, periods=["2020"], start=2020, end=2020)
    msgs = prompt.build_messages(record, table_fields(record), "문서")
    assert msgs[0] == {"role": "system", "content": OLD_SYSTEM}
    assert msgs[1]["content"] == OLD_USER.format(doc="문서", summary=prompt.field_summary(record, table_fields(record)))
