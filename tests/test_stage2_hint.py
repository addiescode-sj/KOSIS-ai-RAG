"""KOS-192: the stage 2 selector's opt-in search phrase (hint) and rewrite path prompt (rerank.hcx).

hint=None (or no argument) must send exactly what the selector sent before KOS-192: the same system prompt (v4) and
the same user message bytes. A hint adds one line `검색용 표현: <hint>` right under the question and nothing else.
No API: the LLM is kosis_rag.llm.base.MockClient, which records (system, user) per call.
"""

import dataclasses
import functools

import pytest
from langchain_core.documents import Document

from kosis_rag.llm.base import MockClient
from kosis_rag.rerank import constants
from kosis_rag.rerank.hcx import HcxSelector, system_prompt, user_message
from kosis_rag.rerank.stage2 import rerank_stage2

pytestmark = pytest.mark.offline

OLD = "질문: 서울 인구\n\n후보 통계표:\n[1] 표 a\n\n[2] 표 b"  # the pre-KOS-192 message (test_no_answer pins it too)
REPLY = '{"picked": [2], "reason": null}'


def _selector(**kw) -> tuple[HcxSelector, MockClient]:
    client = MockClient(REPLY)
    return HcxSelector(text_of={"a": "표 a", "b": "표 b"}, thinking="none", client=client, **kw), client


def test_no_hint_is_the_old_message_byte_for_byte():
    assert user_message("서울 인구", ["표 a", "표 b"]) == OLD
    assert user_message("서울 인구", ["표 a", "표 b"], None) == OLD
    assert user_message("서울 인구", ["표 a", "표 b"], "") == OLD  # an empty hint adds nothing either


def test_a_hint_adds_exactly_one_line_under_the_question():
    with_hint = user_message("서울 인구", ["표 a", "표 b"], "시도별 주민등록인구")
    lines, old = with_hint.split("\n"), OLD.split("\n")
    assert lines[1] == f"{constants.HINT_LABEL}: 시도별 주민등록인구" == "검색용 표현: 시도별 주민등록인구"
    assert lines[:1] + lines[2:] == old and len(lines) == len(old) + 1


def test_a_multiline_hint_is_flattened_to_one_line():
    flat = user_message("서울 인구", ["표 a", "표 b"], "  시도별\n주민등록인구\r\n\t2020년  ")
    lines = flat.split("\n")
    assert lines[1] == "검색용 표현: 시도별 주민등록인구 2020년" and len(lines) == len(OLD.split("\n")) + 1
    assert "\r" not in flat and "\t" not in flat
    assert user_message("서울 인구", ["표 a", "표 b"], " \n\t ") == OLD  # whitespace only: no line


def test_default_selector_sends_the_v4_prompt_and_the_old_message():
    system_prompt.cache_clear()
    select, client = _selector()
    out = select("서울 인구", ["a", "b"])
    assert out.keys == ["b"]
    assert client.calls == [(system_prompt(constants.PROMPT_FILE), OLD)]
    assert select.prompt == constants.PROMPT_FILE == "stage2-hcx-v4.md"
    select("서울 인구", ["a", "b"], hint=None)
    assert client.calls[1] == client.calls[0]  # hint=None: the very same call


def test_rewrite_path_selector_uses_v5_rw_and_the_hint_line():
    system_prompt.cache_clear()
    base, _ = _selector()
    client = MockClient(REPLY)
    select = dataclasses.replace(base, prompt=constants.PROMPT_FILE_RW, client=client)
    docs = rerank_stage2("서울 인구", [], functools.partial(select, hint="주민등록인구"), k_in=20)  # no keys: no call
    assert docs == [] and client.calls == []
    pool = [Document(page_content=k, metadata={"key": k, "score": 0.5}) for k in ("a", "b")]
    docs = rerank_stage2("서울 인구", pool, functools.partial(select, hint="주민등록인구"), k_in=20, name="gpt")
    assert [d.metadata["key"] for d in docs] == ["b", "a"] and docs[0].metadata["stage2_picked_n"] == 1
    system, user = client.calls[0]
    assert system == system_prompt(constants.PROMPT_FILE_RW) != system_prompt(constants.PROMPT_FILE)
    assert user == OLD.replace("\n\n후보", "\n검색용 표현: 주민등록인구\n\n후보", 1)
