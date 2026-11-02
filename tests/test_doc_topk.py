"""scripts/doc_topk_experiment.py (T2-21): chunk scores join their table, RRF, the k curve and the k rule."""

import importlib.util
from pathlib import Path

import numpy as np

from evals.schemas import RetrievalItem

_spec = importlib.util.spec_from_file_location(
    "doc_topk_experiment", Path(__file__).resolve().parents[1] / "scripts" / "doc_topk_experiment.py")
x = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x)


def _item(qid: str, *slots: list[str]) -> RetrievalItem:
    qtype = "multi_table" if len(slots) > 1 else "exact_name"
    return RetrievalItem(qid=qid, split="dev", qtype=qtype, source="scenario", question=qid, required=list(slots))


def test_max_pool_lifts_a_table_by_its_best_chunk_only():
    keys = ["a", "b", "c"]
    doc = np.array([0.5, 0.4, 0.3])
    out = x.max_pool(doc, keys, np.array([0.2, 0.9, 0.35]), ["b", "b", "c"])
    assert out.tolist() == [0.5, 0.9, 0.35]
    assert x.rank_keys(out, keys) == ["b", "a", "c"]
    assert doc.tolist() == [0.5, 0.4, 0.3]  # the document scores are not changed in place


def test_rrf_counts_a_key_once_per_list_at_its_best_rank():
    # chunk lists repeat a table; only its first (best) position counts
    fused = x.rrf([["a", "b", "c"], ["c", "c", "b"]], c=0)
    assert fused[0] == "c"  # 1/3 + 1/1
    assert set(fused) == {"a", "b", "c"}
    assert x.rrf([["a", "b"], ["b", "a"]], c=60) == ["a", "b"]  # ties keep the first list's order


def test_curve_hits_any_slot_and_mrr_stops_at_10():
    items = [_item("q1", ["t1"], ["t2"]), _item("q2", ["t9"])]
    ranked = {"q1": ["x", "t2", "t1"], "q2": [f"y{i}" for i in range(12)] + ["t9"]}
    c = x.curve(items, ranked)
    assert c["hit@1"] == 0 and c["hit@2"] == 1 and c["hit@10"] == 1 and c["hit@15"] == 2
    assert c["mrr@10"] == round((1 / 2) / 2, 4)  # q2 is found at 13, outside MRR@10


def test_show_k_stops_where_extra_k_adds_less_than_one_question_per_five():
    c = {f"hit@{k}": v for k, v in zip(x.KS, (10, 12, 14, 16, 20, 21, 21, 22, 22))}
    # 1->2 +2, 2->3 +2, 3->5 +1/k, 5->10 +0.8/k, 10->15 +0.2/k (not below 0.2), 15->20 0 -> stop at 15
    assert x.show_k(c) == 15
    flat = {f"hit@{k}": 5 for k in x.KS}
    assert x.show_k(flat) == 1
