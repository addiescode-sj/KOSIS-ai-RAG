"""T2-39: the search graph's retrieve node split into parallel list nodes (runs) + fuse.

1. With the question index off, the graph gives what the pre-T2-39 graph gave (master ba72da4's retrieve node,
   unchanged since e7782ed, copied below as _old_state, calling master's hybrid_search frozen verbatim as
   _master_hybrid_search, so a regression in the refactored parts is caught): docs (order and metadata), removed,
   relaxed, route, result.
2. With it on, the hq list shows in rank_hq, moves tables by RRF, and HQ_WEIGHT weighs it; the question is embedded
   once for dense and hq (embed node); question-index hits outside the registry are skipped.
3. The node API's retrieve (make_nodes, T2-37) runs the split steps as one node: its chain equals the graph.

No model, no server, no API: the same kind of fakes as tests/test_graph.py.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from kosis_chat.llm.model import MockClient  # noqa: E402
from kosis_rag import chain  # noqa: E402
from kosis_rag.graph import SearchState, build_graph, make_nodes, run_search  # noqa: E402
from kosis_rag.hcx import HcxResult  # noqa: E402
from kosis_rag.legacy.retrievers import HqRetriever, HybridRetriever, hybrid_search  # noqa: E402
from kosis_rag.parse import node as query_rules  # noqa: E402
from kosis_rag.pipeline import SearchSettings, build_search, load_resources  # noqa: E402
from kosis_rag.rerank import rerank_stage1, rerank_stage2  # noqa: E402
from kosis_rag.retrieve.fuse import rrf_fuse  # noqa: E402
from kosis_rag.retrieve.lists import condition_filter, search_list  # noqa: E402
from kosis_rag.schemas.catalog import TableRecord  # noqa: E402
from kosis_rag.schemas.common import TableInfo  # noqa: E402
from kosis_rag.shared.catalog import to_document  # noqa: E402
from kosis_rag.shared.filters import (TableFilter, filter_tables, filter_with_relaxation, relax, soft_rank,  # noqa: E402
                                      why_excluded)

pytestmark = pytest.mark.offline


def _record(key, name, periods):
    info = TableInfo(
        key=key, org_id="101", tbl_id=key.split("_", 1)[1], survey_name="인구동향조사", organization="국가데이터처",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet=f"values/{key}.parquet", url=f"http://kosis.kr/{key}", menu_path=f"인구 > {name}",
    )
    return TableRecord(info=info, cycle=["Y"], prd_se=["Y"], periods=list(periods), start=int(periods[0][:4]),
                       end=int(periods[-1][:4]), menu_paths=[f"인구 > {name}"])


RECORDS = [_record("101_DT_A", "출생아수", ("2020", "2023")), _record("101_DT_B", "사망자수", ("2023",)),
           _record("101_DT_C", "혼인건수", ("2010",)), _record("101_DT_D", "이혼건수", ("2023",))]
BY_KEY = {r.info.key: r for r in RECORDS}
KEYS = list(BY_KEY)
A, B, C, D = KEYS


class CatalogRetriever(BaseRetriever):
    """Dense / question-index stand-in: fixed keys in order, catalog metadata, `with_filter` like Chroma."""

    keys: list[str]
    records: dict = {}
    last_removed: dict = {}
    allowed: list[str] | None = None

    def with_filter(self, f, records):
        passed, removed = filter_tables(records, f)
        return self.model_copy(update={"allowed": sorted(passed), "last_removed": removed})

    def _get_relevant_documents(self, query, *, run_manager):
        out = []
        for i, key in enumerate(self.keys):
            if self.allowed is None or key in self.allowed:
                doc = to_document(BY_KEY[key].info)
                out.append(Document(page_content=key, metadata={**doc.metadata, "score": 0.9 - i / 10}, id=key))
        return out


class PlainRetriever(BaseRetriever):
    """No `with_filter` (like BM25): filtered after the search."""

    keys: list[str]

    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content=k, metadata={"key": k, "score": 1.0}, id=k) for k in self.keys]


def _stage1(q, keys):
    return [{A: 0.1, B: 0.5, C: 0.9, D: 0.3}[k] for k in keys]


def _pick(*keys, reason=None):
    return lambda q, ks: HcxResult(keys=list(keys), calls=1, usage={"promptTokens": 7}, reason=reason)


def _plain(docs):
    """Documents without the timing fields (they differ between two runs of the same code)."""
    return [(d.id, d.page_content, {k: v for k, v in d.metadata.items() if not k.endswith("_seconds")}) for d in docs]


def _settings(**kw):
    return SearchSettings(tables=Path("unused.jsonl"), **{"cond_filter": False, "stage2": "none", **kw})


def _master_hybrid_search(query: str, retrievers: list[BaseRetriever], *, names: list[str] | None = None,
                          table_filter: TableFilter | None = None, records: dict[str, TableRecord] | None = None,
                          k: int = 50, k_each: int = 50, c: int = 60, score_from: int = 0
                          ) -> tuple[list[Document], dict[str, str]]:
    """master `retriever.hybrid_search` (e7782ed, unchanged in ba72da4), frozen: the body is copied verbatim (only
    the docstring and the name differ), so a regression in the T2-39 split (condition_filter / search_list /
    rrf_fuse) cannot pass by changing both sides of the comparison. Do not update it with the refactored code."""
    # 1. one list per retriever: a retriever with with_filter (Chroma) filters before its search, the others after
    lists: list[list[Document]] = []
    removed: dict[str, str] = {}
    if table_filter is None or table_filter.is_empty():
        lists = [r.invoke(query)[:k_each] for r in retrievers]
        if table_filter is not None and records and table_filter.is_soft():
            lists = [soft_rank(docs, records, table_filter) for docs in lists]  # soft signals only re-rank
    else:
        if not records:
            raise ValueError("a condition needs records (tables.jsonl) to decide on")
        _, dropped, _ = filter_with_relaxation(records, table_filter)
        table_filter = relax(table_filter, dropped)
        for r in retrievers:
            if hasattr(r, "with_filter"):
                filtered: Any = r.with_filter(table_filter, records)
                docs = filtered.invoke(query)
                removed.update(filtered.last_removed)
            else:
                docs = []
                for d in r.invoke(query):
                    record = records.get(d.metadata["key"])
                    reason = why_excluded(record, table_filter) if record is not None else None
                    if reason is None:
                        docs.append(d)
                    else:
                        removed[d.metadata["key"]] = reason
                docs = soft_rank(docs, records, table_filter)
            lists.append(docs[:k_each])

    # 2. fuse by RRF over table keys
    names = names or [f"r{i}" for i in range(len(lists))]
    rrf: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    first: dict[str, Document] = {}
    for name, docs in zip(names, lists):
        for rank, d in enumerate(docs, 1):
            key = d.metadata["key"]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (c + rank)
            ranks.setdefault(key, {})[name] = rank
            first.setdefault(key, d)

    # 3. keep the score_from list's score (the no-answer threshold is calibrated on it, T2-09)
    anchor = lists[score_from] if score_from < len(lists) else []
    own = {d.metadata["key"]: d.metadata.get("score") for d in anchor}
    floor = min((v for v in own.values() if v is not None), default=None)
    order = sorted(rrf, key=lambda key: -rrf[key])  # stable: ties keep first-seen order
    out = []
    for key in order[:k]:
        d = first[key]
        score, source = (own[key], names[score_from]) if key in own else (floor, "other")
        out.append(Document(page_content=d.page_content, id=d.id, metadata={
            **d.metadata, "score": score, "score_from": source, "rrf_score": rrf[key],
            **{f"rank_{n}": r for n, r in ranks[key].items()}}))
    return out, removed


def _old_state(question, resources, client, k=5):
    """What the pre-T2-39 graph (master ba72da4) returned for the same resources: its retrieve node verbatim (parse
    filter, relax, one master hybrid_search over dense + BM25, frozen above), then the same stage and answer functions
    in order."""
    s = resources.settings
    query = query_rules.parse_query(question, sigungu=resources.sigungu)
    table_filter = query.to_filter() if s.cond_filter else None
    relaxed = []
    if table_filter is not None and not table_filter.is_empty():
        _, relaxed, _ = filter_with_relaxation(resources.records, table_filter)
        table_filter = relax(table_filter, relaxed)
    docs, removed = _master_hybrid_search(question, [resources.dense, resources.bm25], names=["dense", "bm25"],
                                          table_filter=table_filter, records=resources.records, k=s.pool,
                                          k_each=s.pool, score_from=0)
    if resources.stage1_score is not None:
        docs = rerank_stage1(question, docs, resources.stage1_score, n=s.rerank_n, mode=s.rerank_mode)
    if resources.select is not None:
        docs = rerank_stage2(question, docs, resources.select, k_in=s.stage2_k, name=resources.stage2_name)
    route, reason = chain.decide_route(docs[:k])
    result = (chain.refusal_answer(question, docs[:k], reason) if route == "refuse"
              else chain.guide_answer(question, docs[:k], client))
    return {"docs": docs, "removed": removed, "relaxed": relaxed, "route": route, "refusal_reason": reason,
            "result": result}


# --- 1. HQ off == the pre-T2-39 graph ---

@pytest.mark.parametrize("question, cond", [
    ("출생아 수", False),
    ("2023년 인구", True),  # the year filter removes DT_C
    ("2023년 성별 인구", True),  # no table has 성: relaxed, the year stays
    ("1990년 인구", True),  # no table holds the year: refusal, nothing relaxed
    ("서울 출생아 수", True),  # a 시도 condition no record can decide on
])
@pytest.mark.parametrize("stages", ["both", "stage1", "none"])
def test_graph_without_hq_equals_the_old_graph(question, cond, stages):
    stage1 = _stage1 if stages in ("both", "stage1") else None
    select = _pick(B) if stages == "both" else None
    reply = "[1] DT_A 표입니다."
    resources = load_resources(_settings(cond_filter=cond), records=RECORDS, dense=CatalogRetriever(keys=[C, A, B, D]),
                               stage1_score=stage1, select=select)
    assert resources.hq is None and resources.hq_bm25 is None
    old_client, new_client = MockClient(reply), MockClient(reply)
    old = _old_state(question, resources, old_client)
    state = run_search(build_graph(resources, new_client), question)
    assert _plain(state.docs) == _plain(old["docs"])
    assert (state.removed, state.relaxed, state.route, state.refusal_reason) == (
        old["removed"], old["relaxed"], old["route"], old["refusal_reason"])
    assert state.result == old["result"] and old_client.calls == new_client.calls
    assert set(state.runs) == {"dense", "bm25"}


def test_graph_without_hq_has_no_question_nodes_even_when_a_fake_is_given():
    # HQ_BUILD empty: an injected question index is not used
    resources = load_resources(_settings(), records=RECORDS, dense=CatalogRetriever(keys=KEYS),
                               hq=CatalogRetriever(keys=[D]), hq_bm25=PlainRetriever(keys=[D]))
    assert resources.hq is None and resources.hq_bm25 is None
    nodes = set(build_graph(resources, MockClient()).get_graph().nodes)
    assert not any(n.startswith("retrieve_hq") for n in nodes)


@pytest.mark.parametrize("f", [None, TableFilter(), TableFilter(years=[2023]), TableFilter(dimensions=["성"]),
                               TableFilter(years=[1990])])
def test_hybrid_search_is_its_parts(f):
    """hybrid_search == condition_filter + search_list per list + rrf_fuse (the graph's condition/retrieve_*/fuse).
    A filter relaxed to nothing (dimensions only) keeps hybrid_search's filtering path."""
    dense, sparse = CatalogRetriever(keys=[C, A, B]), PlainRetriever(keys=[B, D, C])
    docs, removed = hybrid_search("q", [dense, sparse], names=["dense", "bm25"], table_filter=f, records=BY_KEY)
    filtered = f is not None and not f.is_empty()
    g = condition_filter(f, BY_KEY)
    parts = [search_list("q", r, table_filter=g, records=BY_KEY, filtered=filtered) for r in (dense, sparse)]
    assert docs == rrf_fuse([p[0] for p in parts], ["dense", "bm25"])
    assert removed == {**parts[0][1], **parts[1][1]}
    old = HybridRetriever(retrievers=[dense, sparse], names=["dense", "bm25"], records=BY_KEY,
                          condition=(lambda q: f) if f is not None else None)
    assert old.invoke("q") == hybrid_search("q", [dense, sparse], names=["dense", "bm25"], records=BY_KEY,
                                            table_filter=relax(f, old.last_relaxed) if f is not None else None)[0]


@pytest.mark.parametrize("f", [None, TableFilter(), TableFilter(years=[2023]), TableFilter(dimensions=["성"]),
                               TableFilter(years=[1990])])
@pytest.mark.parametrize("k, k_each", [(50, 50), (2, 2), (3, 1)])
def test_hybrid_search_and_hybrid_retriever_equal_master(f, k, k_each):
    """The refactored hybrid_search (and HybridRetriever on it) against master's body, frozen in this file."""
    dense, sparse = CatalogRetriever(keys=[C, A, B]), PlainRetriever(keys=[B, D, C])
    kw = dict(names=["dense", "bm25"], table_filter=f, records=BY_KEY, k=k, k_each=k_each)
    assert hybrid_search("q", [dense, sparse], **kw) == _master_hybrid_search("q", [dense, sparse], **kw)
    old = HybridRetriever(retrievers=[dense, sparse], names=["dense", "bm25"], records=BY_KEY, k=k, k_each=k_each,
                          condition=(lambda q: f) if f is not None else None)
    relaxed = relax(f, filter_with_relaxation(BY_KEY, f)[1]) if f is not None and not f.is_empty() else f
    assert old.invoke("q") == _master_hybrid_search("q", [dense, sparse], **{**kw, "table_filter": relaxed})[0]


def test_rrf_fuse_weights_default_to_plain_rrf():
    lists = [CatalogRetriever(keys=[A, B]).invoke("q"), PlainRetriever(keys=[B, C]).invoke("q")]
    assert rrf_fuse(lists, ["dense", "bm25"]) == rrf_fuse(lists, ["dense", "bm25"], weights=[1.0, 1.0])
    heavy = rrf_fuse(lists, ["dense", "bm25"], weights=[1.0, 3.0])
    assert [d.metadata["key"] for d in heavy] == [B, C, A]  # B: 1/62 + 3/61, C: 3/62, A: 1/61


# --- 2. HQ on ---

def _hq_resources(*, weight=1.0, hq_keys=(D, C), dense_keys=KEYS, bm25=True, cond=False, **kw):
    """Resources with the question index on: dense A..D, BM25 = [A] only (fixed fake), hq = hq_keys."""
    settings = _settings(cond_filter=cond, hq_build="hq-test", hq_weight=weight, hq_bm25=bm25)
    resources = load_resources(settings, records=RECORDS, dense=CatalogRetriever(keys=list(dense_keys)),
                               hq=CatalogRetriever(keys=list(hq_keys)),
                               hq_bm25=PlainRetriever(keys=[D]) if bm25 else None, **kw)
    return replace(resources, bm25=PlainRetriever(keys=[A]))


def _keys(state):
    return [d.metadata["key"] for d in state.docs]


@pytest.mark.parametrize("weight, order", [
    (1.0, [A, D, C, B]),  # A 2/61; D 1/64 + 1/61 > C 1/63 + 1/62 > B 1/62
    (0.5, [A, C, D, B]),  # half weight: C 1/63 + .5/62 > D 1/64 + .5/61
    (2.0, [D, C, A, B]),  # double weight: the question list leads
])
def test_hq_list_is_fused_with_its_weight(weight, order):
    resources = _hq_resources(weight=weight, bm25=False)
    state = run_search(build_graph(resources, MockClient("[1] DT_A 표입니다.")), "출생아 수")
    assert _keys(state) == order
    meta = {d.metadata["key"]: d.metadata for d in state.docs}
    assert meta[D]["rank_hq"] == 1 and meta[C]["rank_hq"] == 2 and "rank_hq" not in meta[A]
    assert all(m["score_from"] == "dense" for m in meta.values())  # the dense cosine stays the score
    assert set(state.runs) == {"dense", "bm25", "hq"} and "retrieve_hq" in state.trace
    assert state.trace["fuse"]["lists"] == ["dense", "bm25", "hq"]


def test_hq_changes_the_order_against_hq_off():
    off = load_resources(_settings(), records=RECORDS, dense=CatalogRetriever(keys=KEYS))
    off = replace(off, bm25=PlainRetriever(keys=[A]))
    before = run_search(build_graph(off, MockClient("[1] DT_A 표입니다.")), "출생아 수")
    after = run_search(build_graph(_hq_resources(bm25=False), MockClient("[1] DT_A 표입니다.")), "출생아 수")
    assert _keys(before) == [A, B, C, D] and _keys(after) == [A, D, C, B]


def test_hq_bm25_adds_its_own_list():
    resources = _hq_resources(bm25=True)
    graph = build_graph(resources, MockClient("[1] DT_A 표입니다."))
    assert {"retrieve_hq", "retrieve_hq_bm25"} <= set(graph.get_graph().nodes)
    state = run_search(graph, "출생아 수")
    meta = {d.metadata["key"]: d.metadata for d in state.docs}
    assert meta[D]["rank_hq_bm25"] == 1 and set(state.runs) == {"dense", "bm25", "hq", "hq_bm25"}
    assert _keys(state)[0] == D  # D: dense 4th + hq 1st + hq_bm25 1st


def test_a_table_only_the_question_index_found_gets_the_floor_score():
    resources = _hq_resources(dense_keys=[A, B], hq_keys=[D], bm25=False)
    state = run_search(build_graph(resources, MockClient("[1] DT_A 표입니다.")), "출생아 수")
    meta = {d.metadata["key"]: d.metadata for d in state.docs}
    assert meta[D]["score_from"] == "other" and meta[D]["score"] == pytest.approx(0.8)  # dense's lowest
    assert meta[D]["table_name"] == "이혼건수"  # catalog metadata comes with the hq hit, so it can be cited


def test_hq_list_gets_the_same_condition_filter():
    resources = _hq_resources(cond=True, bm25=False)
    state = run_search(build_graph(resources, MockClient("[1] DT_A 표입니다.")), "2023년 인구")
    assert C not in [d.metadata["key"] for d in state.runs["hq"]]  # DT_C holds 2010 only
    assert C not in _keys(state) and set(state.removed) == {C}


def test_hq_with_rerankers_runs_them_after_fuse():
    resources = _hq_resources(bm25=False, stage1_score=_stage1, select=_pick(B))
    state = run_search(build_graph(resources, MockClient("[1] DT_B 표입니다.")), "출생아 수")
    assert _keys(state)[0] == B and set(state.trace) >= {"fuse", "rerank1", "rerank2"}


def test_label_and_old_path_ignore_hq_when_off():
    assert _settings().label() == "hybrid"
    assert _settings(hq_build="hq-x", hq_bm25=True).label() == "hybrid+hq+hqbm25"
    assert _settings(hq_build="hq-x", hq_weight=0.5).label() == "hybrid+hq(w0.5)"
    # build_search (nested retrievers) never loads the question index: no Chroma connection is tried here
    old = build_search(_settings(hq_build="hq-x", hq_bm25=True), records=RECORDS, dense=CatalogRetriever(keys=KEYS))
    assert isinstance(old, HybridRetriever) and old.names == ["dense", "bm25"]


def test_tracing_blames_fuse_where_the_pool_is_formed():
    """T2-36 blame on the split graph: the list nodes carry no pool, so a table no list found is lost at fuse."""
    from evals.trace import NodeSpans, lost_at

    resources = _hq_resources(dense_keys=[A, B], hq_keys=[B], bm25=False, stage1_score=_stage1)
    resources = replace(resources, settings=resources.settings.model_copy(update={"rerank_mode": "score"}))
    nodes = NodeSpans()
    run_search(build_graph(resources, MockClient("[1] DT_A 표입니다.")), "출생아 수", {"callbacks": [nodes]})
    keyed = [s["name"] for s in nodes.spans if "keys" in s]
    assert keyed == ["fuse", "rerank1"]
    assert {s["name"] for s in nodes.spans} >= {"retrieve_dense", "retrieve_bm25", "retrieve_hq"}
    node, reason = lost_at(nodes.spans, {D}, 1)
    assert node == "fuse" and "풀(2개)에 없음" in reason
    node, _ = lost_at(nodes.spans, {A}, 1)  # fuse has A first; stage 1 (score mode) puts B (0.5) over A (0.1)
    assert node == "rerank1"


# --- 3. one question embedding for dense and hq (review S4), registry guard (review S5) ---

class EmbeddingRetriever(CatalogRetriever):
    """Dense stand-in that embeds the question itself, like ChromaTableRetriever (an embed_query field)."""

    embed_query: Any = None
    seen: Any = None  # a list: the vector of every search (shared by model copies)

    def _get_relevant_documents(self, query, *, run_manager):
        self.seen.append(self.embed_query(query))
        return super()._get_relevant_documents(query, run_manager=run_manager)


class QuestionCollection:
    """Chroma stand-in for the question index: fixed hits (id, key, text, distance); records the query vectors."""

    def __init__(self, hits):
        self.hits = hits
        self.vectors = []

    def query(self, query_embeddings, n_results, include, where=None):
        self.vectors.append(query_embeddings[0])
        rows = [h for h in self.hits if where is None or h[1] in where["key"]["$in"]][:n_results]
        return {"ids": [[h[0] for h in rows]], "documents": [[h[2] for h in rows]],
                "metadatas": [[{"key": h[1], "type": "concept"} for h in rows]], "distances": [[h[3] for h in rows]]}


class CountingEmbedder:
    """embed_query that counts calls and returns a new vector each time (so a second call would show)."""

    def __init__(self):
        self.calls = []

    def __call__(self, question):
        self.calls.append(question)
        return [float(len(self.calls)), 0.0]


GONE = "101_DT_GONE"  # a table the question index knows but tables.jsonl does not


def _embedding_resources(embedder, collection, seen, **kw):
    dense = EmbeddingRetriever(keys=KEYS, embed_query=embedder, seen=seen)
    hq = HqRetriever(collection=collection, embed_query=embedder, text_of={key: key for key in KEYS}, records=BY_KEY,
                     n_docs=len(collection.hits))
    settings = _settings(hq_build="hq-test", **kw)
    return load_resources(settings, records=RECORDS, dense=dense, hq=hq)


def test_the_question_is_embedded_once_for_dense_and_hq():
    embedder, seen = CountingEmbedder(), []
    collection = QuestionCollection([(f"{D}#q0", D, "이혼 건수", 0.1), (f"{C}#q0", C, "혼인 건수", 0.2)])
    resources = _embedding_resources(embedder, collection, seen)
    graph = build_graph(resources, MockClient("[1] DT_A 표입니다."))
    assert "embed" in graph.get_graph().nodes
    state = run_search(graph, "출생아 수")
    assert embedder.calls == ["출생아 수"]  # one call, from the embed node only
    assert seen == [[1.0, 0.0]] and collection.vectors == [[1.0, 0.0]]  # both lists searched with that vector
    assert state.trace["embed"]["calls"] == 1 and state.query_vecs == {"dense": [1.0, 0.0], "hq": [1.0, 0.0]}
    assert set(state.runs) == {"dense", "bm25", "hq"} and state.runs["hq"][0].metadata["key"] == D
    run_search(graph, "사망자 수")
    assert len(embedder.calls) == 2  # once per question


def test_without_hq_there_is_no_embed_node_and_the_result_is_the_old_one():
    embedder, seen = CountingEmbedder(), []
    resources = load_resources(_settings(cond_filter=True), records=RECORDS,
                               dense=EmbeddingRetriever(keys=[C, A, B, D], embed_query=embedder, seen=seen))
    graph = build_graph(resources, MockClient("[1] DT_A 표입니다."))
    assert "embed" not in graph.get_graph().nodes
    state = run_search(graph, "2023년 인구")
    assert len(embedder.calls) == 1 and "embed" not in state.trace and state.query_vecs == {}
    old = _old_state("2023년 인구", resources, MockClient("[1] DT_A 표입니다."))
    assert _plain(state.docs) == _plain(old["docs"]) and state.removed == old["removed"]


def test_question_index_hits_outside_the_registry_are_skipped_and_the_answer_still_comes():
    embedder, seen = CountingEmbedder(), []
    collection = QuestionCollection([(f"{GONE}#q0", GONE, "없는 표 질문", 0.05), (f"{D}#q0", D, "이혼 건수", 0.1)])
    resources = _embedding_resources(embedder, collection, seen)
    state = run_search(build_graph(resources, MockClient("[1] DT_D 표입니다.")), "이혼 건수")
    assert [d.metadata["key"] for d in state.runs["hq"]] == [D]  # GONE (the best hit) skipped, D kept
    assert GONE not in _keys(state) and state.result is not None
    assert all(c.table_name for c in state.result.candidates)  # every cited table has its catalog fields


# --- 4. the node API's retrieve (T2-37 make_nodes) runs the split steps as one node ---

def _node_chain(resources, client, state):
    """What a node API client gets by calling every node in turn (the answer node by route)."""
    for name, fn in make_nodes(resources, client).items():
        if name in ("refuse", "guide") and name != state.route:
            continue
        state = state.model_copy(update=fn(state))
    return state


@pytest.mark.parametrize("hq", [False, True])
def test_node_api_retrieve_is_the_split_steps_and_the_chain_equals_the_graph(hq):
    def resources():
        embedder, seen = CountingEmbedder(), []
        if hq:
            collection = QuestionCollection([(f"{D}#q0", D, "이혼 건수", 0.1), (f"{C}#q0", C, "혼인 건수", 0.2)])
            return _embedding_resources(embedder, collection, seen, cond_filter=True), embedder, seen
        dense = EmbeddingRetriever(keys=[C, A, B, D], embed_query=embedder, seen=seen)
        return load_resources(_settings(cond_filter=True), records=RECORDS, dense=dense), embedder, seen

    question, reply = "2023년 인구", "[1] DT_A 표입니다."
    graph_resources, _, _ = resources()
    expected = run_search(build_graph(graph_resources, MockClient(reply)), question)
    api_resources, embedder, seen = resources()
    assert list(make_nodes(api_resources, MockClient(reply))) == ["parse", "retrieve", "gate", "refuse", "guide"]
    forged = SearchState(question=question, query_vecs={"dense": [9.0, 9.0]}, runs={"hq": expected.docs[:1]})
    state = _node_chain(api_resources, MockClient(reply), forged)

    assert _plain(state.docs) == _plain(expected.docs) and state.result == expected.result
    assert (state.removed, state.relaxed, state.route) == (expected.removed, expected.relaxed, expected.route)
    assert set(state.runs) == set(expected.runs) and state.query_vecs == {}  # the client's list and vector ignored
    assert len(embedder.calls) == 1 and [9.0, 9.0] not in seen
    lists = ["retrieve_dense", "retrieve_bm25", *(["retrieve_hq"] if hq else [])]
    assert state.trace["retrieve"]["steps"] == ["condition", *(["embed"] if hq else []), *lists, "fuse"]
    assert set(state.trace) >= {"parse", "retrieve", "condition", "fuse", "gate", *lists}
