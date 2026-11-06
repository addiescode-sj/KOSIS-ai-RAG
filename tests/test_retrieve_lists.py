"""T2-41 S2: the retrieve stage (retrieve/) returns what it removed / relaxed instead of keeping it (P2-D17), and the
legacy retrievers (legacy/retrievers.py) that wrap it give what they gave (P2-D18).

No model, no server: the Chroma collections are fakes that return fixed distances.
"""

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from kosis_rag.legacy.retrievers import BM25TableRetriever, ChromaTableRetriever, HqRetriever
from kosis_rag.retrieve import node as retrieve_step
from kosis_rag.retrieve.bm25 import bm25_index
from kosis_rag.retrieve.dense import DenseIndex
from kosis_rag.retrieve.fuse import rrf_fuse
from kosis_rag.retrieve.hq import HqIndex
from kosis_rag.retrieve.lists import condition_filter, search_list
from kosis_rag.retrieve.schemas import FuseInput, ListInput, ListOutput
from kosis_rag.shared.catalog import to_document
from kosis_rag.shared.filters import TableFilter
from tests.test_graph import BY_KEY, KEYS, RECORDS, _settings
from tests.test_hq_index import FakeCollection as FakeQuestions

pytestmark = pytest.mark.offline
A, B, C, D = KEYS
FILTERS = [None, TableFilter(), TableFilter(years=[2023]), TableFilter(dimensions=["성"]), TableFilter(years=[1990])]
TEXT_OF = {key: f"text {key}" for key in KEYS}


class FakeTables:
    """collection.query over every table (distance by position), where = key $in, n_results cuts."""

    def __init__(self, keys=(C, A, B, D)):
        self.keys = list(keys)
        self.calls = []

    def query(self, query_embeddings, n_results, include, where=None):
        self.calls.append({"vector": query_embeddings[0], "n_results": n_results, "where": where})
        keys = [k for k in self.keys if where is None or k in where["key"]["$in"]][:n_results]
        return {"ids": [keys], "documents": [[f"doc {k}" for k in keys]],
                "metadatas": [[to_document(BY_KEY[k].info).metadata for k in keys]],
                "distances": [[0.1 * (i + 1) for i in range(len(keys))]]}


def _embed(calls):
    return lambda q: calls.append(q) or [1.0, 0.0]


@pytest.mark.parametrize("f", FILTERS)
def test_dense_index_equals_the_chroma_retriever(f):
    """DenseIndex.search == ChromaTableRetriever.with_filter(f).invoke(), its last_* coming back as values."""
    index = DenseIndex(collection=FakeTables(), embed_query=_embed([]), k=3)
    legacy = ChromaTableRetriever(collection=FakeTables(), embed_query=_embed([]), k=3)
    scoped = legacy.with_filter(f, BY_KEY) if f is not None else legacy
    docs = scoped.invoke("q")
    assert index.search("q", table_filter=f, records=BY_KEY) == (docs, scoped.last_removed, scoped.last_relaxed, 0)


def test_dense_index_keeps_no_state_and_uses_a_given_vector():
    calls = []
    index = DenseIndex(collection=FakeTables(), embed_query=_embed(calls))
    before = dict(vars(index))
    docs, removed, relaxed, missing = index.search("q", vector=[0.5, 0.5], table_filter=TableFilter(years=[2023]),
                                                   records=BY_KEY)
    assert [d.id for d in docs] == [A, B, D] and set(removed) == {C} and relaxed == [] and missing == 0
    assert calls == [] and index.collection.calls[-1]["vector"] == [0.5, 0.5]  # the embedder was not called
    assert vars(index).keys() == before.keys() and not [n for n in vars(index) if n.startswith("last_")]


@pytest.mark.parametrize("f", FILTERS)
def test_hq_index_equals_the_hq_retriever(f):
    kw = dict(embed_query=lambda q: [0.0], text_of=TEXT_OF, records=BY_KEY, n_docs=5)
    index = HqIndex(collection=FakeQuestions(_hits()), **kw)
    legacy = HqRetriever(collection=FakeQuestions(_hits()), **kw)
    scoped = legacy.with_filter(f, BY_KEY) if f is not None else legacy
    docs = scoped.invoke("q")
    out = index.search("q", table_filter=f, records=BY_KEY)
    assert out == (docs, scoped.last_removed, scoped.last_relaxed, scoped.last_missing)
    assert out[3] == (1 if f is None or f.is_empty() else 0)  # a filter sends key $in: the unknown table is not asked


def _hits():
    gone = "101_DT_GONE"  # a table not in the registry: skipped, counted in missing
    return [(f"{gone}#q0", gone, "없는 표", 0.05), (f"{A}#q0", A, "출생", 0.1), (f"{C}#q0", C, "혼인", 0.2),
            (f"{B}#q1", B, "사망", 0.3)]


def test_bm25_index_equals_the_bm25_retriever():
    docs = [Document(page_content=f"출생아 수 {k}" if k == A else f"사망 {k}", metadata={"key": k}, id=k) for k in KEYS]
    index = bm25_index(docs, k=2)
    assert index.search("출생아") == BM25TableRetriever.from_documents(docs, k=2).invoke("출생아")
    assert BM25TableRetriever.from_index(index).invoke("사망") == index.search("사망")


@pytest.mark.parametrize("f", FILTERS)
def test_search_list_returns_its_state_for_every_kind_of_list(f):
    """(docs, removed, relaxed, missing) for a DenseIndex and the retriever wrapping it, a Bm25Index and its wrapper."""
    g = condition_filter(f, BY_KEY)
    filtered = f is not None and not f.is_empty()
    dense = DenseIndex(collection=FakeTables(), embed_query=_embed([]))
    legacy = ChromaTableRetriever(collection=FakeTables(), embed_query=_embed([]))
    new = search_list("q", dense, table_filter=g, records=BY_KEY, filtered=filtered)
    assert len(new) == 4 and new == search_list("q", legacy, table_filter=g, records=BY_KEY, filtered=filtered)
    texts = [Document(page_content=f"인구 {k}", metadata={**to_document(BY_KEY[k].info).metadata}, id=k) for k in KEYS]
    bm25 = bm25_index(texts, k=10)
    assert (search_list("인구", bm25, table_filter=g, records=BY_KEY, filtered=filtered)
            == search_list("인구", BM25TableRetriever.from_index(bm25), table_filter=g, records=BY_KEY, filtered=filtered))


def test_search_list_hands_the_vector_to_every_kind_of_list():
    calls = []
    for lst in (DenseIndex(collection=FakeTables(), embed_query=_embed(calls)),
                ChromaTableRetriever(collection=FakeTables(), embed_query=_embed(calls))):
        search_list("q", lst, vector=[0.25, 0.75])
        search_list("q", lst, table_filter=TableFilter(years=[2023]), records=BY_KEY, vector=[0.25, 0.75])
        assert lst.collection.calls[-1]["vector"] == [0.25, 0.75]
    assert calls == []


def test_retrieve_steps_are_search_list_and_rrf_fuse():
    from tests.test_graph import CatalogRetriever, _load

    resources = _load(_settings(), records=RECORDS, dense=CatalogRetriever(keys=[C, A, B, D]), sigungu={})
    f = TableFilter(years=[2023])
    runs, removed = {}, {}
    for name, lst, _ in retrieve_step.search_lists(resources):
        out = retrieve_step.retrieve_list(name, ListInput(question="출생아 수", table_filter=f), resources)
        docs, dropped, relaxed, missing = search_list("출생아 수", lst, table_filter=f, records=resources.records,
                                                      k_each=resources.options.pool)
        assert out == ListOutput(docs=docs, removed=dropped, relaxed=relaxed, missing=missing)
        runs[name], removed[name] = out.docs, out.removed
    fused = retrieve_step.fuse(FuseInput(runs=runs, run_removed=removed), resources)
    assert fused.docs == rrf_fuse([runs["dense"], runs["bm25"]], ["dense", "bm25"], k=resources.options.pool)
    assert fused.removed == {**removed["dense"], **removed["bm25"]}
    with pytest.raises(ValidationError):
        ListInput(question="q", k_each=3)
