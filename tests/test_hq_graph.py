"""T2-39: the question generation graph (kosis_rag.hq): cache (keyed by prompt, model and table content), one re-ask,
global rules, the Doc2Query−− filter, the build id (model and filter settings in it) and the files a build writes.

No model, no API: the LLM is a fake with scripted replies, the scorer a fake function, every path under tmp_path.
"""

import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")

from kosis_chat.llm.model import STATUS_OK  # noqa: E402
from kosis_rag.hq import prompt as hq_prompt  # noqa: E402
from kosis_rag.hq.graph import (build_id, build_table_graph, cache_path, filter_scores, model_slug,  # noqa: E402
                                run_build, user_sha)
from kosis_rag.hq.rules import check_questions  # noqa: E402
from kosis_rag.hq.schemas import HqQuestion, HqState  # noqa: E402
from kosis_rag.schemas.catalog import TableClass, TableRecord  # noqa: E402
from kosis_rag.schemas.common import TableInfo  # noqa: E402
from kosis_rag.shared.fields import table_fields  # noqa: E402

pytestmark = pytest.mark.offline

MODEL = "fake-mini"
TYPES6 = ["concept", "concept", "condition", "condition", "distinct", "keyword"]


def _record(key, name, periods=("2015", "2020", "2023"), sido=("서울특별시", "부산광역시", "대구광역시")):
    info = TableInfo(
        key=key, org_id="101", tbl_id=key.split("_", 1)[1], survey_name="인구총조사", organization="국가데이터처",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet=f"values/{key}.parquet", url=f"http://kosis.kr/{key}", menu_path=f"인구 > {name}",
    )
    classes = [TableClass(name="행정구역별", labels=["전국", *sido], n_labels=len(sido) + 1,
                          levels={"전국": 1, "시도": len(sido)}),
               TableClass(name="성별", labels=["남자", "여자"], n_labels=2)]
    return TableRecord(info=info, cycle=["Y"], prd_se=["Y"], periods=list(periods), start=int(periods[0]),
                       end=int(periods[-1]), menu_paths=[f"인구 > 가구 > {name}"], classes=classes, unit="가구")


ONE = _record("101_DT_ONE", "성 및 연령별 1인가구")
TWO = _record("101_DT_TWO", "혼인상태별 인구", periods=("2020",))
RECORDS = {r.info.key: r for r in (ONE, TWO)}
GOOD = ["혼자 사는 20대가 많은 동네", "나 홀로 사는 노인 가구", "2020년 서울 혼자 사는 사람",
        "부산 30대 남성 1인가구", "1인가구가 해마다 어떻게 늘었는지 시도별로", "시도 연령별 1인가구"]


def _reply(texts=GOOD, desc="시도별로 혼자 사는 가구를 성별과 나이대로 나눠 보여 주는 표", types=TYPES6):
    return json.dumps({"description": desc, "questions": [{"type": t, "text": x} for t, x in zip(types, texts)]},
                      ensure_ascii=False)


class FakeLLM:
    """generate(system, user, history) with scripted replies; an Exception in the script is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, user, history=None):
        self.calls.append({"system": system, "user": user, "history": history})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(status=STATUS_OK, text=reply, usage={"input_tokens": 100, "output_tokens": 40},
                               error=None, attempts=1)


def _run(client, tmp_path, key="101_DT_ONE", scorer=None):
    graph = build_table_graph(RECORDS, client, model=MODEL, cache_dir=tmp_path / "cache", scorer=scorer)
    return HqState.model_validate(graph.invoke({"key": key}))


# --- prompt ---

def test_prompt_is_fixed_and_carries_the_table_facts():
    assert hq_prompt.prompt_hash() == hq_prompt.prompt_hash() and len(hq_prompt.prompt_hash()) == 64
    msgs = hq_prompt.build_messages(ONE, table_fields(ONE), "통계표: 성 및 연령별 1인가구")
    assert [m["role"] for m in msgs] == ["system", "user"] and "JSON" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "수록 연도: 2015, 2020, 2023" in user and "서울특별시" in user and "단위: 가구" in user
    assert "{" not in user  # every placeholder filled
    assert hq_prompt.years_text([2015, 2016, 2017]) == "2015~2017 (매년)"


# --- 1. cache and re-ask ---

def test_a_cache_hit_makes_no_call(tmp_path):
    first = FakeLLM(_reply())
    state = _run(first, tmp_path)
    assert len(first.calls) == 1 and state.error is None and len(state.questions) == 6
    user_hash = user_sha(state.messages)
    path = cache_path(tmp_path / "cache", hq_prompt.prompt_hash(), MODEL, "101_DT_ONE", user_hash)
    assert path == tmp_path / "cache" / hq_prompt.prompt_hash()[:8] / MODEL / f"101_DT_ONE-{user_hash[:8]}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert path.is_file() and body["model"] == MODEL and body["user_sha256"] == user_hash
    again = FakeLLM()  # any call would fail: the script is empty
    rerun = _run(again, tmp_path)
    assert again.calls == [] and rerun.questions == state.questions and rerun.responses[0]["cached"] is True


def test_another_model_is_not_a_cache_hit(tmp_path):
    _run(FakeLLM(_reply()), tmp_path)
    other = FakeLLM(_reply())
    graph = build_table_graph(RECORDS, other, model="other-model", cache_dir=tmp_path / "cache")
    graph.invoke({"key": "101_DT_ONE"})
    assert len(other.calls) == 1
    assert {p.parent.name for p in (tmp_path / "cache").rglob("*.json")} == {MODEL, "other-model"}


def test_a_changed_table_is_not_a_cache_hit(tmp_path):
    """Review S1: the same key with other content (here one more year) must not reuse the old reply."""
    _run(FakeLLM(_reply()), tmp_path)
    changed = _record("101_DT_ONE", "성 및 연령별 1인가구", periods=("2015", "2020", "2023", "2024"))
    client = FakeLLM(_reply())
    graph = build_table_graph({**RECORDS, "101_DT_ONE": changed}, client, model=MODEL, cache_dir=tmp_path / "cache")
    graph.invoke({"key": "101_DT_ONE"})
    assert len(client.calls) == 1 and "2024" in client.calls[0]["user"]
    assert len(list((tmp_path / "cache").rglob("101_DT_ONE-*.json"))) == 2  # one file per table content
    unchanged = FakeLLM()  # the old content is still a hit
    assert _run(unchanged, tmp_path).error is None and unchanged.calls == []


def test_a_cache_file_for_another_table_text_is_a_miss(tmp_path):
    """A file at the right path but written for another user message (or by an older version without the hash)."""
    state = _run(FakeLLM(_reply()), tmp_path)
    path = cache_path(tmp_path / "cache", hq_prompt.prompt_hash(), MODEL, "101_DT_ONE", user_sha(state.messages))
    body = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**body, "user_sha256": "0" * 64}), encoding="utf-8")
    client = FakeLLM(_reply())
    assert _run(client, tmp_path).error is None and len(client.calls) == 1


def test_an_invalid_reply_gets_exactly_one_reask(tmp_path):
    client = FakeLLM("죄송합니다, JSON이 아닙니다", _reply())
    state = _run(client, tmp_path)
    assert len(client.calls) == 2 and state.attempts == 2 and state.error is None and len(state.questions) == 6
    second = client.calls[1]
    assert second["history"][-1] == {"role": "assistant", "content": "죄송합니다, JSON이 아닙니다"}
    assert "형식이 맞지 않습니다" in second["user"] and second["system"] == client.calls[0]["system"]
    rerun = FakeLLM()  # both replies are cached
    assert _run(rerun, tmp_path).questions == state.questions and rerun.calls == []


def test_a_reply_still_invalid_after_the_reask_is_an_error(tmp_path):
    bad_type = _reply(types=["concept", "unknown", "condition", "condition", "distinct", "keyword"])
    client = FakeLLM("{}", bad_type)
    state = _run(client, tmp_path)
    assert len(client.calls) == 2 and state.error.startswith("invalid:") and state.questions == []


def test_an_api_failure_is_not_cached(tmp_path):
    state = _run(FakeLLM(TimeoutError("slow")), tmp_path)
    assert state.error.startswith("api: TimeoutError")
    assert not list((tmp_path / "cache").rglob("*.json"))
    assert _run(FakeLLM(_reply()), tmp_path).error is None  # a rerun tries again


# --- 2. rules ---

@pytest.mark.parametrize("text, rule", [
    ("2013년 서울 혼자 사는 사람", "year"),  # the table holds 2015, 2020, 2023
    ("1990년대 부산 1인가구", "year"),
    ("광주 혼자 사는 20대", "sido"),  # 광주 is not one of the table's 시도
    ("강남구 1인가구 3만 가구 맞아", "number"),
    ("소득 5000 이상 1인가구", "number"),  # 4 digits that are not a year: a value even without a unit
    ("1인가구 비율이 12.5%인 곳", "number"),
    ("성 및 연령별 1인가구", "name_copy"),
    ("이 표에서 2020년 자료", "deictic"),
])
def test_rules_drop_what_the_table_does_not_hold(text, rule):
    kept, dropped = check_questions([HqQuestion(type="condition", text=text)], ONE, table_fields(ONE))
    assert kept == [] and dropped[text].startswith(rule)


@pytest.mark.parametrize("text", ["2020년 서울 혼자 사는 사람", "부산 30대 남성 1인가구", "65세 이상 혼자 사는 노인",
                                  "2015, 2020년 대구 1인가구 변화", "시도 연령별 1인가구"])
def test_rules_keep_years_ages_and_held_regions(text):
    kept, dropped = check_questions([HqQuestion(type="condition", text=text)], ONE, table_fields(ONE))
    assert [q.text for q in kept] == [text] and dropped == {}


def test_rules_drop_a_near_duplicate_within_the_table_and_any_sido_for_a_table_without_regions():
    qs = [HqQuestion(type="concept", text="혼자 사는 20대가 많은 동네"),
          HqQuestion(type="concept", text="혼자 사는 20대가 많은 동네는")]
    kept, dropped = check_questions(qs, ONE, table_fields(ONE))
    assert [q.text for q in kept] == ["혼자 사는 20대가 많은 동네"] and dropped["혼자 사는 20대가 많은 동네는"].startswith("duplicate")
    national = _record("101_DT_NAT", "전국 1인가구", sido=())
    national = national.model_copy(update={"classes": national.classes[1:]})  # no administrative class
    kept, dropped = check_questions([HqQuestion(type="condition", text="서울 1인가구")], national, table_fields(national))
    assert kept == [] and dropped["서울 1인가구"].startswith("sido")


def test_a_bare_gwangju_passes_on_a_table_that_holds_gyeonggi():
    """광주 is also 경기도 광주시: dropped where neither 광주광역시 nor 경기도 is held, kept on a 경기도 table."""
    gyeonggi = _record("101_DT_GG", "시군구별 가구", sido=("서울특별시", "부산광역시", "경기도"))
    kept, dropped = check_questions([HqQuestion(type="condition", text="광주 1인가구 추이")], gyeonggi,
                                    table_fields(gyeonggi))
    assert [q.text for q in kept] == ["광주 1인가구 추이"] and dropped == {}
    kept, _ = check_questions([HqQuestion(type="condition", text="광주광역시 1인가구 추이")], gyeonggi,
                              table_fields(gyeonggi))
    assert kept == []  # the full name is the 광역시 only


def test_rule_check_node_keeps_reply_positions(tmp_path):
    texts = ["혼자 사는 20대가 많은 동네", "2013년 서울 1인가구", "2020년 서울 혼자 사는 사람",
             "광주 1인가구", "1인가구가 해마다 어떻게 늘었는지 시도별로", "시도 연령별 1인가구"]
    state = _run(FakeLLM(_reply(texts)), tmp_path)
    assert state.positions == [0, 2, 4, 5] and set(state.dropped) == {texts[1], texts[3]}


# --- 3. score and the global filter ---

def test_score_node_scores_kept_questions_against_their_table(tmp_path):
    seen = []

    def scorer(key, questions):
        seen.append((key, questions))
        return [float(i) for i in range(len(questions))]

    state = _run(FakeLLM(_reply()), tmp_path, scorer=scorer)
    assert seen == [("101_DT_ONE", GOOD)] and state.scores == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def _items(scores):
    return [{"id": f"{key}#q{i}", "key": key, "pos": i, "type": "concept", "text": f"{key} {i}", "score": s}
            for key, ss in scores.items() for i, s in enumerate(ss)]


def test_filter_drops_the_global_bottom_but_keeps_two_per_table():
    items = _items({"T1": [0.1, 0.2, 0.9], "T2": [0.05, 0.06], "T3": [0.5, 0.6, 0.7, 0.8, 0.3]})
    kept, dropped = filter_scores(items, 0.3, 2)
    # bottom 3 of 10 = T2#q0, T2#q1, T1#q0; T2 has only two, so both stay
    assert [it["id"] for it in dropped] == ["T1#q0"]
    assert {it["key"] for it in kept} == {"T1", "T2", "T3"} and len(kept) == 9
    kept, dropped = filter_scores(_items({"T1": [0.1, 0.2, 0.9, 0.95], "T3": [0.5, 0.6, 0.7, 0.8, 0.3, 0.4]}), 0.3, 2)
    assert [it["id"] for it in dropped] == ["T1#q0", "T1#q1", "T3#q4"]  # round(0.3 * 10) = 3, none protected


def test_filter_keeps_everything_without_scores_or_with_frac_zero():
    items = _items({"T1": [0.1, 0.2, 0.3]})
    assert filter_scores(items, 0.0, 2) == (items, [])
    unscored = [{**it, "score": None} for it in items]
    assert filter_scores(unscored, 0.3, 2) == (unscored, [])


def test_build_id_is_deterministic_and_names_model_and_filter():
    a = build_id("a" * 64, "b" * 64, "gpt-4o-mini")
    assert a == build_id("a" * 64, "b" * 64, "gpt-4o-mini") == "hq-aaaaaaaa-bbbbbbbb-gpt-4o-mini-f30-k2"
    assert build_id("a" * 64, "b" * 64, "gpt-4o-mini", 0.0) == "hq-aaaaaaaa-bbbbbbbb-gpt-4o-mini-f0"  # no filter, no k
    assert build_id("a" * 64, "b" * 64, "gpt-4o-mini", limit=20).endswith("-f30-k2-n20")
    # review S2: another model or min_keep is another build (folder and Chroma collection)
    assert build_id("a" * 64, "b" * 64, "gpt-4o") != a
    assert build_id("a" * 64, "b" * 64, "gpt-4o-mini", min_keep=3) == "hq-aaaaaaaa-bbbbbbbb-gpt-4o-mini-f30-k3"
    assert build_id("a" * 64, "b" * 64, "gpt-4o-mini", 0.25) == "hq-aaaaaaaa-bbbbbbbb-gpt-4o-mini-f25-k2"
    assert model_slug("GPT 4.1/Mini") == "gpt-4-1-mini" and model_slug("???") == "model"
    assert re.fullmatch(r"[a-z0-9-]+", a) and re.fullmatch(r"[a-z0-9-]+", build_id("a" * 64, "b" * 64, "x.y_z"))


# --- 4. a whole build ---

def test_run_build_writes_questions_report_and_resumes_from_the_cache(tmp_path):
    two = ["혼인 안 한 사람이 얼마나 되나", "결혼한 사람 비율 변화", "2020년 서울 미혼 인구",
           "대구 미혼 남성", "2020년 인구총조사 혼인상태 시도별", "시도 혼인상태 인구"]
    client = FakeLLM(_reply(), _reply(two, desc="혼인상태별 인구를 시도별로 보여 주는 표"))
    scores = {"101_DT_ONE": [0.9, 0.1, 0.8, 0.7, 0.6, 0.5], "101_DT_TWO": [0.2, 0.3, 0.95, 0.4, 0.85, 0.75]}
    scorer = lambda key, qs: scores[key][: len(qs)]  # noqa: E731
    kw = dict(model=MODEL, tables_sha256="c" * 64, scorer=scorer, cache_dir=tmp_path / "cache",
              out_root=tmp_path / "hq", concurrency=1, log=lambda s: None)
    state = run_build([ONE, TWO], client, **kw)
    assert len(client.calls) == 2 and state.build_id == f"hq-{hq_prompt.prompt_hash()[:8]}-cccccccc-fake-mini-f30-k2"
    rows = [json.loads(line) for line in (tmp_path / "hq" / state.build_id / "questions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    # 12 scored, bottom round(3.6) = 4: ONE#q1 (0.1), TWO#q0 (0.2), TWO#q1 (0.3), TWO#q3 (0.4)
    assert {r["id"] for r in rows} == {f"101_DT_ONE#q{i}" for i in (0, 2, 3, 4, 5)} | {
        f"101_DT_TWO#q{i}" for i in (2, 4, 5)}
    assert set(rows[0]) == {"id", "key", "type", "text"} and state.totals["score_dropped"] == 4
    report = (tmp_path / "hq" / state.build_id / "report.md").read_text(encoding="utf-8")
    assert "## 유형별" in report and "seed 0" in report and state.build_id in report
    descs = (tmp_path / "hq" / state.build_id / "descriptions.jsonl").read_text(encoding="utf-8")
    assert "혼인상태별 인구를 시도별로" in descs
    again = FakeLLM()
    rerun = run_build([ONE, TWO], again, **kw)
    assert again.calls == [] and rerun.build_id == state.build_id and rerun.totals["calls"] == 0
    assert rerun.totals["cached"] == 2
    build = json.loads((tmp_path / "hq" / state.build_id / "build.json").read_text(encoding="utf-8"))
    assert build["model"] == MODEL and build["min_keep"] == 2 and build["filter_frac"] == 0.3


def test_builds_with_another_model_or_min_keep_do_not_overwrite_each_other(tmp_path):
    """Review S2: each model / min_keep gets its own data/hq/<build_id>/ (and so its own Chroma collection)."""
    scorer = lambda key, qs: [0.5] * len(qs)  # noqa: E731
    kw = dict(tables_sha256="f" * 64, scorer=scorer, cache_dir=tmp_path / "cache", out_root=tmp_path / "hq",
              concurrency=1, log=lambda s: None)
    a = run_build([ONE], FakeLLM(_reply()), model=MODEL, **kw)
    b = run_build([ONE], FakeLLM(_reply()), model="other-mini", **kw)
    c = run_build([ONE], FakeLLM(), model=MODEL, min_keep=3, **kw)  # same model: the cache serves it
    assert len({a.build_id, b.build_id, c.build_id}) == 3
    for st, model, keep in ((a, MODEL, 2), (b, "other-mini", 2), (c, MODEL, 3)):
        build = json.loads((tmp_path / "hq" / st.build_id / "build.json").read_text(encoding="utf-8"))
        assert (build["model"], build["min_keep"]) == (model, keep)


def test_run_build_without_scorer_keeps_every_rule_passing_question(tmp_path):
    state = run_build([ONE], FakeLLM(_reply()), model=MODEL, tables_sha256="d" * 64, limit=1,
                      cache_dir=tmp_path / "cache", out_root=tmp_path / "hq", log=lambda s: None)
    assert state.build_id.endswith("-fake-mini-f0-n1")
    assert state.totals["kept"] == 6 and state.totals["score_dropped"] == 0


def test_run_build_reports_a_failed_table_and_goes_on(tmp_path):
    client = FakeLLM(TimeoutError("down"), _reply())
    state = run_build([ONE, TWO], client, model=MODEL, tables_sha256="e" * 64, cache_dir=tmp_path / "cache",
                      out_root=tmp_path / "hq", concurrency=1, log=lambda s: None)
    assert state.totals["failed"] == 1 and state.totals["tables_with_questions"] == 1
    failed = (tmp_path / "hq" / state.build_id / "failed.jsonl").read_text(encoding="utf-8")
    assert "101_DT_ONE" in failed and "TimeoutError" in failed
