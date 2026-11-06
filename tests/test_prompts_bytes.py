"""T2-41 S3 (P2-D22): the prompts moved to .md files are the bytes of the old string constants.

OLD_SYSTEM is the old kosis_rag.hcx.SYSTEM expression copied verbatim (MAX_PICK = 5, as the old module had it), and
tests/fixtures/prompts/stage2-hcx-v3.old-system.txt is the value that module produced, written before it was deleted.
Kept for the moving step only (plans/T2-41.md S3: removed in the next step). The refusal sentences moved as Python
constants (chain.py -> gate/texts.py); their old values are copied here too.

T2-41 S5: the guide prompt (chain.PROMPT, a ChatPromptTemplate of a system and a human string) moved to
backend/app/respond/prompts/guide-v1.md (system, the separator line, human). OLD_GUIDE_SYSTEM / OLD_GUIDE_HUMAN are
the old expressions copied verbatim (NO_MATCH = "NO_MATCH"); kept for this moving step only.
"""

import hashlib
from importlib import resources
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from backend.app.respond import prompts as guide_prompts
from kosis_rag.gate import texts
from kosis_rag.rerank import constants, hcx

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "src" / "kosis_rag" / "rerank" / "prompts" / "stage2-hcx-v3.md"
SNAPSHOT = ROOT / "tests" / "fixtures" / "prompts" / "stage2-hcx-v3.old-system.txt"
OLD_SHA256 = "78fd05e900d47535762a0df621999c4e1f20758e86ad16e3ffa839d70f611d55"  # of the old SYSTEM, UTF-8

MAX_PICK = 5
OLD_SYSTEM = (
    "당신은 KOSIS 인구통계 통계표 검색의 2차 선택기입니다. 사용자의 질문에 답하는 데 쓸 수 있는 후보 통계표를 고르세요. "
    "질문에 직접 답하거나 질문을 다시 쓰지 마세요.\n\n"
    "# 입력 형식\n"
    "질문: 사용자의 원래 질문\n"
    "후보 통계표:\n"
    "[1] 첫 번째 통계표 설명\n"
    "[2] 두 번째 통계표 설명\n"
    "...\n\n"
    "# 판단 단계\n"
    "1. 질문에서 찾으려는 통계 지표와 세는 대상·항목을 파악하세요.\n"
    "2. 각 후보 통계표 설명에 그 지표가 담겨 있는지 확인하세요. 주제만 비슷하고 지표가 다르면 고르지 마세요. "
    "설명에 없는 지표가 있다고 추측하지 마세요.\n"
    f"3. 지표가 맞는 표를 질문과 가까운 순서로 최대 {MAX_PICK}개 고르세요. "
    "출력에는 입력에 나온 후보 번호만 사용하고 같은 번호를 반복하지 마세요.\n"
    "4. 맞는 표가 없으면 사유를 정하세요. 질문이 인구·가구·출생·사망·혼인·이동 등 인구통계 범위 밖이면 "
    "out_of_scope, 인구통계 질문이지만 후보에 필요한 지표가 없으면 no_item입니다.\n"
    "5. 출력 형식을 확인하세요. 표를 하나라도 골랐으면 reason은 null, 고른 표가 없으면 picked는 []이고 "
    "reason은 두 사유 중 하나입니다.\n\n"
    "# 판단 범위\n"
    "연도·기간, 월·분기·연 주기, 시도·시군구 등 세부 지역 단위가 맞는지는 판단하지 마세요. "
    "이 조건은 별도 코드가 확인합니다. 지표가 맞으면 해당 조건이 달라도 표를 고르세요.\n\n"
    "# 출력 형식\n"
    "설명이나 Markdown 없이 JSON 객체 한 줄만 출력하세요.\n"
    '{"picked": [후보 번호, ...], "reason": null}\n'
    "또는\n"
    '{"picked": [], "reason": "out_of_scope"}\n'
    "또는\n"
    '{"picked": [], "reason": "no_item"}\n\n'
    "# 예시\n"
    "입력:\n"
    "질문: 2025년 출생아 수는?\n"
    "후보 통계표:\n"
    "[1] 출생아 수, 2020~2024년\n"
    "[2] 합계출산율, 2020~2024년\n"
    "출력:\n"
    '{"picked": [1], "reason": null}\n\n'
    "입력:\n"
    "질문: 전입자 수는?\n"
    "후보 통계표:\n"
    "[1] 출생아 수\n"
    "[2] 사망자 수\n"
    "출력:\n"
    '{"picked": [], "reason": "no_item"}\n\n'
    "입력:\n"
    "질문: 자동차 등록 대수는?\n"
    "후보 통계표:\n"
    "[1] 주민등록인구\n"
    "출력:\n"
    '{"picked": [], "reason": "out_of_scope"}'
)

OLD_NO_MATCH = "NO_MATCH"
OLD_GUIDE_SYSTEM = (
    "당신은 KOSIS 통계표 안내 도우미입니다. 아래 후보 통계표 목록만 근거로, 질문에 맞는 표를 1~3개 골라 왜 맞는지 짧게 설명하세요.\n"
    "규칙: 목록에 없는 표·기간·숫자를 지어내지 마세요. 통계 수치는 말하지 마세요. 표를 가리킬 때는 [번호]와 표 ID를 쓰세요.\n"
    "고른 표만 [번호]와 표 ID로 가리키고, 고르지 않은 표는 언급하지 마세요.\n"
    f"맞는 표가 하나도 없으면 다른 말 없이 {OLD_NO_MATCH} 만 출력하세요."
)
OLD_GUIDE_HUMAN = "후보 통계표:\n{context}\n\n질문: {question}"
GUIDE_MD = ROOT / "backend" / "app" / "respond" / "prompts" / "guide-v1.md"
OLD_REFUSAL_TEXT = "제공된 KOSIS 통계표 중에서 이 질문에 맞는 표를 찾지 못했습니다. 다른 표현으로 다시 질문해 주세요."
OLD_REFUSALS = {
    "out_of_scope": "이 질문은 제공된 KOSIS 인구 통계표가 다루는 범위 밖입니다. 다른 통계(예: 지역·경제·보건 통계)는 KOSIS에서 직접 찾아 주세요.",
}


def test_old_constant_and_snapshot_agree():
    assert SNAPSHOT.read_bytes() == OLD_SYSTEM.encode("utf-8")
    assert hashlib.sha256(OLD_SYSTEM.encode("utf-8")).hexdigest() == OLD_SHA256


def test_stage2_prompt_file_is_the_old_system_bytes():
    raw = MD.read_bytes()
    assert raw == OLD_SYSTEM.encode("utf-8")
    assert b"\r" not in raw and not raw.endswith(b"\n")  # LF only, no trailing newline (the constant had none)


def test_loader_reads_the_same_text_through_importlib_resources():
    hcx.system_prompt.cache_clear()
    assert hcx.system_prompt() == OLD_SYSTEM
    assert (resources.files("kosis_rag.rerank") / "prompts" / constants.PROMPT_FILE).read_text(encoding="utf-8") == OLD_SYSTEM
    assert constants.PROMPT_FILE == MD.name and constants.MAX_PICK == MAX_PICK


def test_refusal_texts_moved_unchanged():
    assert texts.NO_MATCH == OLD_NO_MATCH
    assert texts.REFUSAL_TEXT == OLD_REFUSAL_TEXT
    assert texts.REFUSALS == OLD_REFUSALS


def test_guide_prompt_file_is_the_old_prompt_bytes():
    raw = GUIDE_MD.read_bytes()
    assert raw == (OLD_GUIDE_SYSTEM + guide_prompts.SEPARATOR + OLD_GUIDE_HUMAN).encode("utf-8")
    assert b"\r" not in raw and not raw.endswith(b"\n")  # LF only, no trailing newline (the strings had none)
    assert raw.count(guide_prompts.SEPARATOR.encode("utf-8")) == 1


def test_guide_loader_gives_the_old_messages_and_template():
    guide_prompts.guide_messages.cache_clear()
    guide_prompts.guide_prompt.cache_clear()
    assert guide_prompts.guide_messages() == (OLD_GUIDE_SYSTEM, OLD_GUIDE_HUMAN)
    old = ChatPromptTemplate.from_messages([("system", OLD_GUIDE_SYSTEM), ("human", OLD_GUIDE_HUMAN)])
    new = guide_prompts.guide_prompt()
    assert new == old and new.input_variables == ["context", "question"]
    values = {"question": "q", "context": "[1] DT_A | 표"}
    assert new.invoke(values).to_messages() == old.invoke(values).to_messages()
    assert texts.NO_MATCH in OLD_GUIDE_SYSTEM  # the prompt asks for the sentinel the guide reads
