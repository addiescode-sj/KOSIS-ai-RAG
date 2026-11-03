"""T2-21: Hit@k curve and the pre-registered k rule (scripts/topk_curve.py)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("topk_curve", Path(__file__).parents[1] / "scripts" / "topk_curve.py")
topk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(topk)


def test_curve_counts_ranks_up_to_k_and_skips_misses():
    c = topk.curve([1, 3, 3, None, 20, 21], k_max=20)
    assert (c[1], c[2], c[3], c[19], c[20]) == (1, 1, 3, 3, 4)


def test_pick_k_is_the_smallest_k_reaching_the_share_of_hit_at_max():
    c = {1: 5, 2: 8, 3: 9, 4: 10}  # threshold 0.9 x 10 = 9
    k, threshold, boundary = topk.pick_k(c)
    assert (k, threshold) == (3, 9.0)
    assert boundary  # 9 - 9.0 < 1: report k and k+1


def test_pick_k_not_boundary_when_a_full_question_above():
    k, _, boundary = topk.pick_k({1: 2, 2: 10, 3: 10})  # threshold 9
    assert k == 2 and not boundary


def test_summarize_uses_answerable_rows_of_the_split_and_skips_qids():
    rows = [
        {"qid": "a", "split": "dev", "answerable": True, "rank": 1},
        {"qid": "b", "split": "test", "answerable": True, "rank": 2},
        {"qid": "c", "split": "test", "answerable": False, "rank": None},
        {"qid": "d", "split": "test", "answerable": True, "rank": None},
    ]
    assert topk.summarize(rows)["n"] == 3
    assert topk.summarize(rows, "test")["curve"][2] == 1
    assert topk.summarize(rows, skip=frozenset({"d"}))["n"] == 2
    assert topk.summarize(rows)["mrr10"] == (1 + 0.5) / 3
