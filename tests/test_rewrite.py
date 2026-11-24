"""KOS-192 steps 1-2: the query rewriter (one LLM call, its reply, R1 / R2 messages, errors), the rules that throw a
rewrite away, and the prompt's own examples run through the real parse step.

No model, no server, no API: the LLM is kosis_rag.llm.base.MockClient (or a client that raises); parsed questions are
built by hand so each rule is checked on its own, except for the prompt examples (parse.node.parse_question).
"""

import re
from types import SimpleNamespace

import pytest

from kosis_rag.llm.base import STATUS_API_ERROR, LLMResult, MockClient
from kosis_rag.llm.config import LLMConfig
from kosis_rag.parse.node import parse_question
from kosis_rag.parse.schemas import ParsedQuery, ParseInput
from kosis_rag.rewrite import constants
from kosis_rag.rewrite.node import (check_rewrite, normalize, parse_reply, rewrite_client, rewrite_question,
                                    system_prompt, user_message)
from kosis_rag.rewrite.schemas import RewriteInput, RewriteOutput

pytestmark = pytest.mark.offline

OK = '{"status": "ok", "rewrite": "2021년 부산 이혼 건수"}'
NONE = '{"status": "none", "rewrite": ""}'


def _q(**kw) -> ParsedQuery:
    return ParsedQuery(**kw)


# --- the call and its messages --------------------------------------------------------------------------------------

def test_r1_message_is_the_question_only():
    assert user_message("2021년 부산에서 갈라선 부부가 몇 쌍이야") == "질문: 2021년 부산에서 갈라선 부부가 몇 쌍이야"
    assert user_message("q", []) == "질문: q"


def test_r2_message_lists_the_table_names_in_order():
    assert user_message("q", ["점유형태별 가구 - 시도", "가구원수별 가구 - 시도"]) == (
        "질문: q\n참고 후보 통계표:\n- 점유형태별 가구 - 시도\n- 가구원수별 가구 - 시도")


def test_r2_message_is_cut_to_max_table_names():
    names = [f"표 {n}" for n in range(constants.MAX_TABLE_NAMES + 2)]
    lines = user_message("q", names).split("\n")
    assert lines[2:] == [f"- 표 {n}" for n in range(constants.MAX_TABLE_NAMES)] and constants.MAX_TABLE_NAMES == 10


def test_the_call_sends_the_prompt_and_the_r1_or_r2_message():
    client = MockClient([OK, OK])
    rewrite_question(RewriteInput(question="q"), client)
    rewrite_question(RewriteInput(question="q", table_names=["A", "B"]), client)
    assert client.calls[0] == (system_prompt(), "질문: q")
    assert client.calls[1] == (system_prompt(), "질문: q\n참고 후보 통계표:\n- A\n- B")
    assert system_prompt().startswith("당신은 KOSIS 인구통계 통계표 검색의 질문 재작성기입니다.")


def test_an_ok_reply_is_used_with_its_tokens():
    client = MockClient([LLMResult(text=OK, usage={"input_tokens": 900, "output_tokens": 20})])
    out = rewrite_question(RewriteInput(question="2021년 부산에서 갈라선 부부가 몇 쌍이야"), client)
    assert isinstance(out, RewriteOutput)
    assert (out.rewrite, out.status, out.reason) == ("2021년 부산 이혼 건수", "used", None)
    assert (out.input_tokens, out.output_tokens) == (900, 20) and out.seconds >= 0


def test_hcx_usage_names_are_read_too():
    client = MockClient([LLMResult(text=OK, usage={"promptTokens": 7, "completionTokens": 3})])
    out = rewrite_question(RewriteInput(question="q"), client)
    assert (out.input_tokens, out.output_tokens) == (7, 3)


def test_a_none_reply_has_no_rewrite():
    out = rewrite_question(RewriteInput(question="2022년 시도별 혼인 건수"), MockClient([NONE]))
    assert (out.rewrite, out.status, out.reason) == (None, "none", None)


def test_the_first_json_object_is_read_in_either_key_order():
    text = '설명입니다 {"status": "ok", "rewrite": " 2019년 경기도 전입자 수 "} {"status": "ok", "rewrite": "x"}'
    assert parse_reply(text) == ("2019년 경기도 전입자 수", "ok")
    assert parse_reply('{"rewrite": "전입자 수", "status": "ok"}') == ("전입자 수", "ok")  # the old order
    assert parse_reply(NONE) == (None, "none") and parse_reply('{"rewrite": "", "status": "none"}') == (None, "none")
    assert parse_reply('{"rewrite": "전입자 수"}') == ("전입자 수", "ok")  # no status, but a rewrite: ok


@pytest.mark.parametrize("reply", ["재작성할 수 없습니다", '{"status": "maybe", "rewrite": "a"}', '{"status": "ok", "rewrite": 3}',
                                   '{"status": "ok"', '{"rewrite": ""}'])
def test_an_unreadable_reply_is_an_error_not_an_exception(reply):
    out = rewrite_question(RewriteInput(question="q"), MockClient([reply]))
    assert out.status == "error" and out.rewrite is None and out.reason
    with pytest.raises(ValueError):
        parse_reply(reply)


def test_a_failed_call_is_an_error_not_an_exception():
    class Boom:
        def generate(self, system, user):
            raise ConnectionError("server down")

    out = rewrite_question(RewriteInput(question="q"), Boom())
    assert (out.status, out.rewrite) == ("error", None) and out.reason == "ConnectionError: server down"
    failed = rewrite_question(RewriteInput(question="q"), MockClient([LLMResult(text="", status=STATUS_API_ERROR,
                                                                              error="503")]))
    assert failed.status == "error" and "503" in failed.reason


def test_an_ok_reply_with_an_empty_rewrite_is_left_to_the_check():
    out = rewrite_question(RewriteInput(question="q"), MockClient(['{"status": "ok", "rewrite": "  "}']))
    assert (out.status, out.rewrite) == ("used", "")
    assert check_rewrite("q", out.rewrite, _q(), _q()) == "empty"


def test_the_client_runs_at_temperature_0_256_tokens_and_a_timeout():
    config = LLMConfig(provider="openai", model="gemma", base_url="http://localhost:8000/v1", api_key="k",
                       temperature=0.7, max_tokens=512)
    client = rewrite_client(config)
    assert (client.temperature, client.max_tokens) == (constants.TEMPERATURE, constants.MAX_TOKENS) == (0.0, 256)
    assert (client.model, client.config.base_url, client.config.timeout) == ("gemma", "http://localhost:8000/v1", 30.0)
    assert rewrite_client(config.model_copy(update={"timeout": 10.0})).config.timeout == 10.0  # a set timeout stays


def test_the_input_is_a_strict_boundary_model():
    with pytest.raises(Exception):
        RewriteInput(question="q", tables=["A"])  # extra field refused


# --- check_rewrite --------------------------------------------------------------------------------------------------

def test_a_good_rewrite_passes():
    a = _q(years=[2021], period_start="2021", period_end="2021", regions=["부산광역시"], measures=["이혼"])
    b = _q(years=[2021], period_start="2021", period_end="2021", regions=["부산광역시"], measures=["이혼"],
           dimensions=["시도"])  # an added dimension is allowed
    assert check_rewrite("2021년 부산에서 갈라선 부부가 몇 쌍이야", "2021년 부산 이혼 건수", a, b) is None


@pytest.mark.parametrize("rewrite", ["", "   ", None])
def test_empty(rewrite):
    assert check_rewrite("q", rewrite, _q(), _q()) == "empty"


def test_too_long():
    assert check_rewrite("질문", "가" * (constants.MAX_CHARS + 1), _q(), _q()) == "too_long"
    assert check_rewrite("질문", "가" * constants.MAX_CHARS, _q(), _q()) is None


def test_unchanged_after_whitespace_punctuation_and_particles():
    assert check_rewrite("2020년 서울 인구, 알려줘?", " 2020년  서울 인구 알려줘", _q(), _q()) == "unchanged"
    assert check_rewrite("2020년 서울의 인구는?", "2020년 서울 인구", _q(), _q()) == "unchanged"
    assert check_rewrite("부산에서 이혼 건수를", "부산 이혼 건수", _q(), _q()) == "unchanged"
    assert normalize("A-b  c.") == "abc" and normalize("서울의 인구는") == "서울인구"


def test_a_new_or_changed_time_number():
    assert check_rewrite("21년 출생아 수", "2021년 출생아 수", _q(), _q()) == "new_number"  # 2021 is new
    assert check_rewrite("인구", "2020년 인구", _q(), _q(years=[2020])) == "new_number"
    assert check_rewrite("출생아", "출생아 3월", _q(), _q()) == "new_number"
    assert check_rewrite("2020년 출생아 수", "2020 출생아 수 통계", _q(), _q()) is None  # the same year, bare


def test_a_dropped_time_number():
    assert check_rewrite("21년 출생아 수", "출생아 수 통계", _q(), _q()) == "dropped_number"
    assert check_rewrite("최근 5년 혼인", "최근 혼인 건수", _q(recent=True), _q(recent=True)) == "dropped_number"
    assert check_rewrite("22년 대전 사별", "22년 대전 혼인상태별 인구(사별)", _q(regions=["대전광역시"]),
                         _q(regions=["대전광역시"])) is None  # "22년" kept as written


def test_numbers_are_compared_after_nfkc_separators_and_quarters():
    assert check_rewrite("２０２１년 출생", "2021년 출생아 수", _q(), _q()) is None
    assert check_rewrite("인구 1,000명 이상 시군구", "인구 1000명 이상 시군구 수", _q(), _q()) is None
    assert check_rewrite("2023년 2/4분기 출생", "2023년 2분기 출생아 수", _q(), _q()) is None


def test_a_number_of_a_classification_term_may_be_added():
    for rewrite in ("2020년 1인가구 수", "2020년 2인 이상 가구 수", "2020년 65세 이상 인구", "2020년 15~64세 인구",
                    "2020년 5세별 인구", "2020년 20대 인구"):
        assert check_rewrite("2020년 혼자 사는 사람이나 젊은 사람 수", rewrite, _q(), _q()) is None, rewrite


def test_any_other_new_number_is_rejected():
    assert check_rewrite("출생아 수", "출생아 수 상위 10개", _q(), _q()) == "new_number"
    assert check_rewrite("가구", "1000대 가구", _q(), _q()) == "new_number"  # not a two-digit 대


def test_years_differ_missing_or_added():
    assert check_rewrite("2020년 인구", "2020 인구 수", _q(years=[2020]), _q()) == "condition_years"
    assert check_rewrite("2020년 인구", "2020년 인구 2020", _q(years=[2020]), _q(years=[2020, 2021])) == "condition_years"
    a = _q(years=[2020, 2021])
    assert check_rewrite("2020년과 2021년 인구", "2021, 2020년 인구수", a, _q(years=[2021, 2020])) is None  # a set


def test_period_differs():
    a = _q(period_start="2015", period_end="2020")
    assert check_rewrite("2015년부터 2020년 인구", "2015년부터, 2020 인구 추이", a,
                         _q(period_start="2015")) == "condition_period"


def test_cycle_kept_or_added_but_never_dropped_or_changed():
    assert check_rewrite("월별 출생", "출생아 수", _q(cycle="M"), _q()) == "condition_cycle"
    assert check_rewrite("연도별 출생", "월별 출생아 수", _q(cycle="Y"), _q(cycle="M")) == "condition_cycle"
    assert check_rewrite("달마다 출생", "월별 출생아 수", _q(), _q(cycle="M")) is None  # added by the rewrite


def test_recent_differs():
    assert check_rewrite("작년 출생", "출생아 수", _q(recent=True), _q()) == "condition_recent"
    assert check_rewrite("출생", "최근 출생아 수", _q(), _q(recent=True)) == "condition_recent"


def test_regions_differ_missing_or_added():
    a = _q(regions=["서울특별시", "부산광역시"])
    assert check_rewrite("서울과 부산 인구", "서울 인구 수", a, _q(regions=["서울특별시"])) == "condition_regions"
    assert check_rewrite("인구", "서울 인구 수", _q(), _q(regions=["서울특별시"])) == "condition_regions"
    assert check_rewrite("서울과 부산 인구", "부산 서울 인구수", a, _q(regions=["부산광역시", "서울특별시"])) is None


def test_reasons_come_in_the_documented_order():
    assert constants.REJECT_REASONS[:5] == ("empty", "too_long", "unchanged", "new_number", "dropped_number")
    # a new time number that also changes the years is reported as the number
    assert check_rewrite("인구", "2020년 인구", _q(), _q(years=[2020])) == "new_number"


# --- the prompt's own examples ----------------------------------------------------------------------------------------

def _examples() -> list[tuple[str, str]]:
    """(question, reply JSON) of every example in the prompt (the part after "# 예시")."""
    body = system_prompt().split("# 예시", 1)[1]
    found = re.findall(r"질문: (.+)\n(?:.*\n)*?출력:\n(\{.*\})", body)
    assert len(found) == 9
    return found


def test_every_ok_example_of_the_prompt_passes_the_check_through_the_real_parser():
    resources = SimpleNamespace(sigungu={})  # parse_question reads only resources.sigungu (tests/test_parse_node.py)
    parse = lambda text: parse_question(ParseInput(question=text), resources)  # noqa: E731
    ok = [(q, parse_reply(reply)[0]) for q, reply in _examples() if parse_reply(reply)[1] == "ok"]
    assert len(ok) == 7
    for question, rewrite in ok:
        assert check_rewrite(question, rewrite, parse(question), parse(rewrite)) is None, question


def test_the_none_examples_of_the_prompt_parse_as_none():
    none = [reply for _, reply in _examples() if '"none"' in reply]
    assert len(none) == 2 and all(parse_reply(reply) == (None, "none") for reply in none)
