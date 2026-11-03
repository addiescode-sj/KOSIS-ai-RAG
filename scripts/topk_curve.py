"""T2-21 (KOS-83): Hit@k curve (k = 1..20) and the pre-registered k rule from L1 runs.

Reads `results.jsonl` of `run_eval --layer l1 --k 20` runs and counts answerable questions by their gold `rank`.
k rule (plans/T2-21.md, fixed before the run): the smallest k with Hit@k >= 0.9 x Hit@20 over dev+test.

    python scripts/topk_curve.py evals/reports/<run> [evals/reports/<run> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

K_MAX = 20
SHARE = 0.9
SHOW = (1, 2, 3, 4, 5, 7, 10, 15, 20)


def curve(ranks: list[int | None], k_max: int = K_MAX) -> dict[int, int]:
    """Hit@k for k = 1..k_max: how many questions have their gold table at rank <= k."""
    return {k: sum(1 for r in ranks if r is not None and r <= k) for k in range(1, k_max + 1)}


def pick_k(c: dict[int, int], share: float = SHARE) -> tuple[int, float, bool]:
    """The smallest k with Hit@k >= share x Hit@k_max.

    Out: (k, threshold, boundary). boundary = Hit@k is less than one question above the threshold,
    so the plan reports k and k+1 together.
    """
    k_max = max(c)
    threshold = share * c[k_max]
    k = next(k for k in sorted(c) if c[k] >= threshold)
    return k, threshold, c[k] - threshold < 1


def mrr10(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= 10) / len(ranks)


def load(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict], split: str = "all", skip: frozenset[str] = frozenset()) -> dict:
    """Curve, k rule and MRR@10 for the answerable rows of one split (skip = qids to leave out)."""
    ranks = [r["rank"] for r in rows
             if r["answerable"] and (split == "all" or r["split"] == split) and r["qid"] not in skip]
    c = curve(ranks)
    k, threshold, boundary = pick_k(c)
    return {"n": len(ranks), "curve": c, "k": k, "threshold": threshold, "boundary": boundary, "mrr10": mrr10(ranks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    head = " | ".join(f"@{k}" for k in SHOW)
    print(f"| run | split | n | {head} | MRR@10 | k (기준) |")
    print("|" + " --- |" * (len(SHOW) + 5))
    for run in args.runs:
        rows = load(run)
        failed = frozenset(r["qid"] for r in rows if r["answerable"] and r.get("stage2_error"))
        variants = [("all", frozenset()), ("dev", frozenset()), ("test", frozenset())]
        if failed:
            variants.append(("all-err", failed))
        for split, skip in variants:
            s = summarize(rows, "all" if split == "all-err" else split, skip)
            hits = " | ".join(str(s["curve"][k]) for k in SHOW)
            mark = " 경계" if s["boundary"] else ""
            print(f"| {run.name} | {split} | {s['n']} | {hits} | {s['mrr10']:.3f} | {s['k']} ({s['threshold']:.1f}){mark} |")
        if any(r.get("stage2_calls") is not None or r.get("stage2_error") for r in rows):
            print(f"  stage2_error (답 있음): {len(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
