"""T2-39: the question index (kosis_rag.hq.index): max-pool per table, the condition filter, the question BM25,
collection names, and how load_resources picks them up.

No model, no server: the Chroma collection is a fake that returns fixed distances.
"""

import json
from pathlib import Path

import pytest
from langchain_core.retrievers import BaseRetriever

from kosis_rag.hq import index as hq_index
from kosis_rag.hq.index import HqRetriever, ckpt8, collection_name, hq_bm25, max_pool, read_questions
from kosis_rag.index import COLLECTION_NAME
from kosis_rag.pipeline import SearchSettings, load_resources
from kosis_rag.retriever import BM25TableRetriever
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.schemas.common import TableInfo
from kosis_rag.shared.filters import TableFilter

pytestmark = pytest.mark.offline


def _record(key, name, periods):
    info = TableInfo(
        key=key, org_id="101", tbl_id=key.split("_", 1)[1], survey_name="인구동향조사", organization="국가데이터처",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet=f"values/{key}.parquet", url=f"http://kosis.kr/{key}", menu_path=f"인구 > {name}",
    )
    return TableRecord(info=info, cycle=["Y"], prd_se=["Y"], periods=list(periods), start=int(periods[0][:4]),
                       end=int(periods[-1][:4]), menu_paths=[f"인구 > {name}"])


RECORDS = [_record("101_DT_A", "1인가구", ("2020", "2023")), _record("101_DT_B", "사망자수", ("2023",)),
           _record("101_DT_C", "혼인건수", ("2010",))]
BY_KEY = {r.info.key: r for r in RECORDS}
A, B, C = BY_KEY
TEXT_OF = {k: f"통계표: {BY_KEY[k].info.table_name}" for k in BY_KEY}
# (question id, table key, text, cosine distance to the query)
HITS = [(f"{A}#q0", A, "혼자 사는 사람", 0.10), (f"{B}#q0", B, "사망 원인", 0.20), (f"{A}#q3", A, "1인가구 추이", 0.30),
        (f"{C}#q1", C, "결혼 건수", 0.40), (f"{B}#q2", B, "사망자 수", 0.50)]


class FakeCollection:
    """collection.query over fixed hits: where = key $in, n_results cuts, best (smallest distance) first."""

    def __init__(self, hits=HITS):
        self.hits = hits
        self.calls = []

    def query(self, query_embeddings, n_results, include, where=None):
        self.calls.append({"n_results": n_results, "where": where})
        rows = [h for h in self.hits if where is None or h[1] in where["key"]["$in"]]
        rows = sorted(rows, key=lambda h: h[3])[:n_results]
        return {"ids": [[h[0] for h in rows]], "documents": [[h[2] for h in rows]],
                "metadatas": [[{"key": h[1], "type": "concept"} for h in rows]], "distances": [[h[3] for h in rows]]}


def _retriever(**kw):
    return HqRetriever(collection=FakeCollection(), embed_query=lambda q: [0.0], text_of=TEXT_OF, records=BY_KEY,
                       n_docs=len(HITS), **kw)


def test_max_pool_keeps_each_tables_best_question():
    pooled = max_pool([("a1", "A", "x", 0.5), ("b1", "B", "y", 0.7), ("a2", "A", "z", 0.9)])
    assert [(t["key"], t["score"], t["best_id"], t["n"]) for t in pooled] == [("A", 0.9, "a2", 2), ("B", 0.7, "b1", 1)]


def test_retriever_returns_tables_with_their_n2_text_and_best_question():
    docs = _retriever().invoke("혼자 사는 사람 많은 곳")
    assert [d.metadata["key"] for d in docs] == [A, B, C]
    a = docs[0]
    assert a.page_content == TEXT_OF[A] and a.id == A
    assert a.metadata["score"] == pytest.approx(0.9) and a.metadata["hq_n"] == 2
    assert a.metadata["hq_best_question"] == "혼자 사는 사람" and a.metadata["hq_best_id"] == f"{A}#q0"
    assert a.metadata["table_name"] == "1인가구" and a.metadata["url"]  # catalog metadata for citing


def test_retriever_asks_for_at_most_n_questions_and_cuts_to_k():
    r = _retriever(n_questions=2)
    docs = r.invoke("q")
    assert r.collection.calls[-1]["n_results"] == 2 and [d.metadata["key"] for d in docs] == [A, B]
    assert [d.metadata["key"] for d in _retriever(k=1).invoke("q")] == [A]
    capped = HqRetriever(collection=FakeCollection(), embed_query=lambda q: [0.0], text_of=TEXT_OF, records=BY_KEY,
                         n_docs=3)
    capped.invoke("q")
    assert capped.collection.calls[-1]["n_results"] == 3  # never more than the collection holds


def test_retriever_respects_the_condition_filter():
    r = _retriever().with_filter(TableFilter(years=[2023]), BY_KEY)
    docs = r.invoke("q")
    assert [d.metadata["key"] for d in docs] == [A, B] and set(r.last_removed) == {C}
    assert sorted(r.collection.calls[-1]["where"]["key"]["$in"]) == [A, B]
    none = _retriever().with_filter(TableFilter(years=[1990]), BY_KEY)
    assert none.invoke("q") == [] and none.collection.calls == []  # no table holds the year: no query at all


def test_hits_and_questions_of_tables_outside_the_registry_are_skipped(caplog):
    """Review S5: an index built from another tables.jsonl must not put a table without catalog fields in the pool."""
    gone = "101_DT_GONE"
    hits = [(f"{gone}#q0", gone, "없는 표 질문", 0.05), *HITS]
    r = HqRetriever(collection=FakeCollection(hits), embed_query=lambda q: [0.0], text_of=TEXT_OF, records=BY_KEY,
                    n_docs=len(hits))
    with caplog.at_level("WARNING", logger="kosis_rag.hq.index"):
        docs = r.invoke("q")
    assert [d.metadata["key"] for d in docs] == [A, B, C] and r.last_missing == 1
    assert all(d.metadata["table_name"] for d in docs) and "not in tables.jsonl" in caplog.text
    one = r.model_copy(update={"k": 1})
    assert [d.metadata["key"] for d in one.invoke("q")] == [A]  # the top k is taken after the skip
    rows = [{"id": f"{gone}#q0", "key": gone, "type": "concept", "text": "없는 표 질문"},
            {"id": f"{A}#q0", "key": A, "type": "concept", "text": "혼자 사는 사람"}]
    with caplog.at_level("WARNING", logger="kosis_rag.hq.index"):
        bm25 = hq_bm25(rows, BY_KEY, TEXT_OF)
    assert [d.id for d in bm25.docs] == [A] and "question BM25" in caplog.text


def test_question_bm25_has_one_document_per_table_with_its_n2_text():
    rows = [{"id": f"{A}#q0", "key": A, "type": "concept", "text": "혼자 사는 사람"},
            {"id": f"{A}#q1", "key": A, "type": "keyword", "text": "1인 가구 추이"},
            {"id": f"{B}#q0", "key": B, "type": "concept", "text": "사망 원인"},
            {"id": f"{C}#q0", "key": C, "type": "concept", "text": "결혼 건수"}]
    bm25 = hq_bm25(rows, BY_KEY, TEXT_OF, k=10)
    assert isinstance(bm25, BM25TableRetriever) and [d.id for d in bm25.docs] == [A, B, C]
    docs = bm25.invoke("혼자 사는 사람이 많아")
    assert [d.metadata["key"] for d in docs] == [A] and docs[0].page_content == TEXT_OF[A]
    assert docs[0].metadata["table_name"] == "1인가구"
    assert [d.id for d in hq_bm25(rows[2:], BY_KEY, TEXT_OF).docs] == [B, C]  # a table without questions is left out


def test_collection_names_follow_chroma_rules():
    name = collection_name("hq-abcd1234-ef567890-f30", "unseen-e1.pt@0123456789ab")
    assert name == "kosis_hq__hq-abcd1234-ef567890-f30__01234567" and COLLECTION_NAME.match(name)
    odd = collection_name("HQ Build!", "x")
    assert COLLECTION_NAME.match(odd) and odd == odd.lower() and " " not in odd
    assert ckpt8("base") == "base" and ckpt8("unseen-e1.pt@0123456789ab") == "01234567"
    assert collection_name("hq-a-b-f30-desc", "base").endswith("-desc__base")


def test_read_questions_and_descriptions(tmp_path):
    build = tmp_path / "hq-x"
    build.mkdir()
    (build / "questions.jsonl").write_text(json.dumps({"id": f"{A}#q0", "key": A, "type": "concept", "text": "q"},
                                                      ensure_ascii=False) + "\n", encoding="utf-8")
    (build / "descriptions.jsonl").write_text(json.dumps({"key": A, "description": "설명"}, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    assert read_questions("hq-x", tmp_path) == [{"id": f"{A}#q0", "key": A, "type": "concept", "text": "q"}]
    assert read_questions("hq-x-desc", tmp_path) == [{"id": f"{A}#d", "key": A, "type": "description", "text": "설명"}]


class ListDense(BaseRetriever):
    """Dense stand-in without a collection (load_resources only needs a retriever here)."""

    def _get_relevant_documents(self, query, *, run_manager):
        return []


def test_load_resources_builds_the_question_bm25_only_when_asked(monkeypatch):
    rows = [{"id": f"{A}#q0", "key": A, "type": "concept", "text": "혼자 사는 사람"}]
    monkeypatch.setattr(hq_index, "read_questions", lambda build, root=None: rows)
    fake_hq = _retriever()

    def settings(**kw):
        return SearchSettings(tables=Path("unused.jsonl"), cond_filter=False, stage2="none", **kw)

    on = load_resources(settings(hq_build="hq-t", hq_bm25=True), records=RECORDS, dense=ListDense(), hq=fake_hq)
    assert on.hq is fake_hq and isinstance(on.hq_bm25, BM25TableRetriever) and [d.id for d in on.hq_bm25.docs] == [A]
    dense_only = load_resources(settings(hq_build="hq-t"), records=RECORDS, dense=ListDense(), hq=fake_hq)
    assert dense_only.hq is fake_hq and dense_only.hq_bm25 is None
    off = load_resources(settings(), records=RECORDS, dense=ListDense(), hq=fake_hq)
    assert off.hq is None and off.hq_bm25 is None
