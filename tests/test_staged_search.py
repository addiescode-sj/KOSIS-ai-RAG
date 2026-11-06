"""T2-06, T2-07: n2 text, hybrid RRF, stage 1 / stage 2 re-rankers, CLOVA parsing and the assembled pipeline.

No model, no server, no API: scores and picks come from fake functions, the Chroma side from a list retriever.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from kosis_rag.adapters.models import resolve_ckpt
from kosis_rag.llm.clova_rerank import WRAP, ClovaReranker, ClovaResult
from kosis_rag.pipeline import SearchSettings, build_search
from kosis_rag.legacy.rerankers import CrossEncoderReranker, LLMReranker
from kosis_rag.legacy.retrievers import HybridRetriever
from kosis_rag.rerank.stage1 import blend_order
from kosis_rag.schemas.catalog import TableClass, TableRecord
from kosis_rag.schemas.common import TableInfo
from kosis_rag.shared import tabletext as text
from kosis_rag.shared.filters import TableFilter

offline = pytest.mark.offline


def _record(key, name, periods=("2020",), classes=(), items=(), levels=None, unit=None):
    info = TableInfo(
        key=key, org_id="101", tbl_id=key.split("_", 1)[1], survey_name="인구동향조사", organization="국가데이터처",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet=f"values/{key}.parquet", url=f"http://kosis.kr/{key}", menu_path=f"인구 > {name}",
    )
    return TableRecord(info=info, cycle=["Y"], prd_se=["Y"], periods=list(periods), start=int(periods[0][:4]),
                       end=int(periods[-1][:4]), items=list(items), menu_paths=[f"인구 > 출생 > {name}"],
                       unit=unit, classes=[TableClass(name=n, labels=list(ls), n_labels=len(ls),
                                                      levels=(levels or {}).get(n, {})) for n, ls in classes])


class ListRetriever(BaseRetriever):
    """Returns fixed keys in order with fixed scores; `with_filter` keeps only the keys that pass (like Chroma)."""

    keys: list[str]
    scores: list[float] = []
    records: dict = {}
    last_removed: dict = {}
    allowed: list[str] | None = None

    def with_filter(self, f, records):
        from kosis_rag.shared.filters import filter_tables

        passed, removed = filter_tables(records, f)
        return self.model_copy(update={"allowed": sorted(passed), "last_removed": removed})

    def _get_relevant_documents(self, query, *, run_manager):
        out = []
        for i, key in enumerate(self.keys):
            if self.allowed is None or key in self.allowed:
                score = self.scores[i] if self.scores else 1.0 - i / 100
                out.append(Document(page_content=key, metadata={"key": key, "score": score}, id=key))
        return out


class PlainRetriever(BaseRetriever):
    """No `with_filter` (like BM25): the hybrid filters its list after the search."""

    keys: list[str]

    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content=k, metadata={"key": k, "score": 1.0}, id=k) for k in self.keys]


def _keys(docs):
    return [d.metadata["key"] for d in docs]


# --- n2 text (must match the text the models were trained on) ---

@offline
def test_n2_name_rules():
    assert text.norm_name("성/연령별 인구 - 시군구") == ("성별 연령별 인구", {"지역 수준": "시군구"})
    assert text.norm_name("혼인건수 (2015~2020)") == ("혼인건수", {"기간": "2015~2020"})
    assert text.norm_name("인구(20세 이상, 일반가구)") == ("인구", {"대상": "20세 이상, 일반가구"})
    assert text.norm_name("출생아수(발생기준")[0] == "출생아수"  # lost closing paren put back, then peeled off
    assert text.norm_query("15세이상  인구") == "15세 이상 인구"


@offline
def test_n2_doc_summarizes_long_label_lists_and_drops_units():
    rec = _record("101_DT_A", "읍면동별 인구", classes=[("행정구역별", [f"동{i}" for i in range(61)])],
                  items=["인구수 (명)", "내국인_남자"])
    doc = text.build_doc(rec, "n2-L60")
    assert "항목: 인구수, 내국인 남자" in doc["embed_text"]
    assert "행정구역별: 61개 (예: 동0, 동1, 동2, 동3, 동4)" in doc["embed_text"]
    assert "동60" in doc["bm25_text"] and "동60" not in doc["embed_text"]
    assert "세는 대상" not in doc["embed_text"] and "단위" not in doc["embed_text"]
    assert text.build_doc(rec, "n2-L60") == text.build_doc(rec.model_dump(), "n2-L60")


@offline
def test_n3_adds_roles_units_and_region_levels():
    sido = ["서울특별시", "부산광역시", "대구광역시"]
    rec = _record("101_DT_B", "시도/성/연령별 혼인건수", classes=[("행정구역별", ["전국", *sido]), ("성별", ["남자", "여자"])],
                  items=["혼인건수 (건)"], levels={"행정구역별": {"전국": 1, "시도": 3}})
    doc = text.build_doc(rec, "n3-L60")  # deprecated format, still readable
    lines = doc["embed_text"].splitlines()
    assert lines[0] == "통계표: 시도별 성별 연령별 혼인건수"
    assert lines[1] == "세는 대상: 혼인건수" and lines[2] == "분류: 시도·성·연령"
    assert "단위: 건" in lines and "행정구역별: 지역 단위 전국·시도(3)" in lines
    assert "서울특별시" not in doc["embed_text"] and "서울특별시" in doc["bm25_text"]  # place names stay in BM25
    assert "성별: 남자, 여자" in lines
    with pytest.raises(ValueError):
        text.build_doc(rec, "n9-L60")


# --- hybrid ---

@offline
def test_rrf_order_and_dense_score_kept():
    dense = ListRetriever(keys=["a", "b", "c"], scores=[0.9, 0.8, 0.7])
    sparse = ListRetriever(keys=["b", "d", "a"])
    docs = HybridRetriever(retrievers=[dense, sparse], names=["dense", "bm25"]).invoke("q")
    assert _keys(docs) == ["b", "a", "d", "c"]  # b: 1/62+1/61, a: 1/61+1/63, d: 1/62, c: 1/63
    by = {d.metadata["key"]: d.metadata for d in docs}
    assert by["b"]["score"] == 0.8 and by["b"]["rank_dense"] == 2 and by["b"]["rank_bm25"] == 1
    assert by["d"]["score"] == 0.7 and by["d"]["score_from"] == "other"  # BM25-only: the lowest dense cosine


@offline
def test_rrf_tie_keeps_first_seen_order_and_cuts_lists():
    docs = HybridRetriever(retrievers=[ListRetriever(keys=["a", "x"]), ListRetriever(keys=["b", "y"])],
                           k_each=1).invoke("q")
    assert _keys(docs) == ["a", "b"]


@offline
def test_condition_filters_both_lists():
    records = {r.info.key: r for r in [_record("101_DT_OLD", "인구", periods=("2010",)),
                                       _record("101_DT_NEW", "인구", periods=("2023",))]}
    dense = ListRetriever(keys=["101_DT_OLD", "101_DT_NEW"])
    sparse = PlainRetriever(keys=["101_DT_OLD", "101_DT_NEW"])
    hybrid = HybridRetriever(retrievers=[dense, sparse], records=records, condition=lambda q: TableFilter(years=[2023]))
    assert _keys(hybrid.invoke("2023년 인구")) == ["101_DT_NEW"]
    assert "101_DT_OLD" in hybrid.last_removed


# --- stage 1 ---

@offline
def test_blend_order_mixes_both_ranks():
    # re-ranker ranks 3,2,1: 0 and 2 get 1/61 + 1/63 (> 1/62 + 1/62 of 1) and tie, so the input order decides
    assert blend_order([0.1, 0.5, 0.9]) == [0, 2, 1]
    # re-ranker ranks 1,3,2: 1 and 2 tie on 1/62 + 1/63, input order again
    assert blend_order([0.9, 0.1, 0.5]) == [0, 1, 2]


@offline
def test_stage1_reorders_top_n_and_keeps_tail():
    base = ListRetriever(keys=["a", "b", "c", "d"], scores=[0.9, 0.8, 0.7, 0.6])
    scores = {"a": 0.1, "b": 0.2, "c": 0.9}
    docs = CrossEncoderReranker(base=base, score=lambda q, ks: [scores[k] for k in ks], n=3, mode="score").invoke("q")
    assert _keys(docs) == ["c", "b", "a", "d"]
    c = docs[0].metadata
    assert (c["input_rank"], c["stage1_rank"], c["score"]) == (3, 1, 0.7)  # the hybrid score stays


@offline
def test_stage1_empty_input():
    assert CrossEncoderReranker(base=ListRetriever(keys=[]), score=lambda q, ks: []).invoke("q") == []


# --- stage 2 ---

@offline
def test_stage2_puts_picked_first_in_stage1_order_and_drops_unknown_keys():
    base = ListRetriever(keys=["a", "b", "c", "d"])
    docs = LLMReranker(base=base, select=lambda q, ks: ["c", "zzz", "b"], k_in=3).invoke("q")
    assert _keys(docs) == ["b", "c", "a", "d"]
    assert docs[0].metadata["stage2_picked"] and docs[0].metadata["stage2_picked_n"] == 2
    assert not docs[2].metadata["stage2_picked"]


@offline
def test_stage2_failure_keeps_order_and_records_error():
    def boom(q, ks):
        raise TimeoutError("slow")

    docs = LLMReranker(base=ListRetriever(keys=["a", "b"]), select=boom).invoke("q")
    assert _keys(docs) == ["a", "b"]
    assert docs[0].metadata["stage2_error"] == "TimeoutError: slow" and docs[0].metadata["stage2_picked_n"] == 0


@offline
def test_stage2_nothing_picked_is_a_signal_not_a_reorder():
    docs = LLMReranker(base=ListRetriever(keys=["a", "b"]),
                       select=lambda q, ks: ClovaResult(keys=[], usage={"promptTokens": 10})).invoke("q")
    assert _keys(docs) == ["a", "b"]
    assert docs[0].metadata["stage2_picked_n"] == 0 and docs[0].metadata["stage2_input_tokens"] == 10


# --- CLOVA response handling (fake transport) ---

@offline
def test_clova_wraps_question_maps_ids_and_retries():
    sent = []
    replies = [{"status": {"code": "42901"}},
               {"status": {"code": "20000"}, "result": {"citedDocuments": [{"id": "2"}, {"id": "9"}, {"id": "2"}],
                                                        "usage": {"promptTokens": 5}}}]

    def send(body, key, timeout):
        sent.append(body)
        return replies[len(sent) - 1]

    select = ClovaReranker(text_of={"a": "A", "b": "B"}, key="k", waits=(0,), send=send)
    out = select("서울 인구", ["a", "b"])
    assert out.keys == ["b"] and out.calls == 2 and out.usage == {"promptTokens": 5}
    assert sent[0]["query"] == WRAP.format(q="서울 인구")
    assert sent[0]["documents"] == [{"id": "1", "doc": "A"}, {"id": "2", "doc": "B"}]


@offline
def test_clova_non_retry_error_raises_without_key_in_message():
    select = ClovaReranker(text_of={"a": "A"}, key="secret-key", send=lambda b, k, t: {"http": 401})
    with pytest.raises(RuntimeError) as err:
        select("q", ["a"])
    assert "secret-key" not in str(err.value)


# --- pipeline ---

@offline
def test_pipeline_assembles_hybrid_stage1_stage2(tmp_path):
    records = [_record("101_DT_A", "출생아수"), _record("101_DT_B", "사망자수"), _record("101_DT_C", "혼인건수")]
    dense = ListRetriever(keys=["101_DT_A", "101_DT_B", "101_DT_C"])
    settings = SearchSettings(tables=tmp_path / "unused.jsonl", cond_filter=False, stage2="none")
    retriever = build_search(settings, records=records, dense=dense,
                             stage1_score=lambda q, ks: [{"101_DT_A": 0.1, "101_DT_B": 0.2, "101_DT_C": 0.9}[k] for k in ks],
                             select=lambda q, ks: ["101_DT_B"])
    docs = retriever.invoke("혼인 건수")
    assert _keys(docs)[0] == "101_DT_B"
    meta = docs[0].metadata
    assert {"rrf_score", "stage1_score", "stage2_picked"} <= set(meta)


@offline
def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("HQ_BUILD", "")  # empty, so load_dotenv (override=False) keeps .env HQ_BUILD out
    monkeypatch.setenv("COND_FILTER", "off")
    monkeypatch.setenv("STAGE2", "CLOVA")
    monkeypatch.setenv("RERANK_CKPT", "")
    s = SearchSettings.from_env()
    assert (s.cond_filter, s.stage2, s.rerank_ckpt) == (False, "clova", None)
    assert s.label() == "hybrid -> clova"


@offline
def test_checkpoint_text_format_is_checked(tmp_path):
    import json

    from kosis_rag.adapters.models import check_format, trained_format

    old, new = tmp_path / "unseen-e2.pt", tmp_path / "seen-e1.pt"
    new.with_suffix(".json").write_text(json.dumps({"doc_format": "n3-L60"}), encoding="utf-8")
    assert trained_format(old) == "n2-L60" and trained_format(new) == "n3-L60" and trained_format(None) is None
    check_format(new, "n3-L60")
    check_format(None, "n3-L60")  # the base model has no training text
    with pytest.raises(ValueError):
        check_format(old, "n3-L60")


@offline
def test_default_text_is_n2_and_n3_is_deprecated():
    rec = _record("101_DT_B", "시도/성/연령별 혼인건수", classes=[("행정구역별", ["전국", "서울특별시"])],
                  items=["혼인건수 (건)"], levels={"행정구역별": {"전국": 1, "시도": 1}})
    assert text.DOC_FORMAT == "n2-L60" and "n3-L60" in text.DEPRECATED and text.DOC_FORMAT not in text.DEPRECATED
    assert text.build_doc(rec) == text.build_doc(rec, "n2-L60")
    assert "세는 대상" not in text.build_doc(rec)["embed_text"]


@offline
def test_search_text_format_follows_collection(tmp_path):
    from kosis_rag.pipeline import text_format

    dense = ListRetriever(keys=[])
    assert text_format(SearchSettings(), dense) == "n2-L60"  # no format on the retriever: default
    s_n2 = SearchSettings(text_format="n2-L60")
    assert text_format(s_n2, dense) == "n2-L60"
    built = type("Built", (), {"doc_format": "n2-L60"})()
    assert text_format(SearchSettings(), built) == "n2-L60"
    with pytest.raises(ValueError):
        text_format(SearchSettings(text_format="n3-L60"), built)


@offline
def test_resolve_ckpt_picks_last_epoch(tmp_path):
    for n in (1, 2, 10):
        (tmp_path / f"unseen-e{n}.pt").write_bytes(b"x")
    assert resolve_ckpt(tmp_path).name == "unseen-e10.pt"
    assert resolve_ckpt(None) is None and resolve_ckpt("") is None
    with pytest.raises(FileNotFoundError):
        resolve_ckpt(tmp_path / "missing.pt")
    (tmp_path / "seen-e2.pt").write_bytes(b"x")
    with pytest.raises(ValueError):  # two regimes in one folder: name the file
        resolve_ckpt(tmp_path)
