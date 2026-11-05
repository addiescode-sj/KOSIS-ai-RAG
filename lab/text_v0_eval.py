"""Plan 53: does the table text matter before any training? Untrained bge-m3 (V0) hybrid search over text variants.

Variants (2 x 2): text format n2-L60 / n3-L60 (kosis_rag.text) x tables.jsonl before the rebuild (no region
levels, 10/29) / after it (levels filled, 11/03). Only the n3 text changes with levels (region classes become
"지역 단위 전국·시도(17)·시군구(250)" instead of a label list), so the four cells separate the format from the data.

Search is the lab V0: dense top-50 (bge-m3, no fine-tuning) + BM25 top-50 on the same variant's texts, RRF c=60.
Questions are normalized the same way for every variant. Reported per variant: routing_v0 dev / test and
retrieval_golden all / dev / test (answerable rows; test was looked at before, so exploratory), hybrid Hit@k and
MRR@10, dense-only and BM25-only Hit@5, and the hybrid Hit@5 difference against n2-L60@old with 95% bootstrap
intervals (questions, and tables as clusters). No golden is used for anything but scoring.

    python lab/text_v0_eval.py                        # old = data/tables_old_1029.jsonl, new = data/tables.jsonl
    python lab/text_v0_eval.py --docs-only            # text statistics per variant, no model, no golden
Writes data/lab/text_v0_eval_<run id>_summary.md and _ranks.jsonl. Document vectors are cached in data/lab/emb/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    DEFAULT_MODEL, DEPTH, METRICS, OUT_DIR, CACHE, RETRIEVAL, ROOT, ROUTING, bm25_tokenize, boot, encode, group_rank,
    model_for, query_text, question_vectors, read_jsonl, rrf, summary, top,
)
from kosis_rag import text as T  # noqa: E402  (lab_common puts src/ on sys.path)

FORMATS = ("n2-L60", "n3-L60")
TABLES = {"old": ROOT / "data" / "tables_old_1029.jsonl", "new": ROOT / "data" / "tables.jsonl"}
BASELINE = "n2-L60@old"
LONG_LIST = re.compile(r": \d+개 \(예: ")


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def variants(paths: dict[str, Path]) -> dict[str, dict]:
    """name -> {docs: [{key, embed, bm25}], sha: tables sha256} for every format x tables file."""
    out = {}
    for tag, path in paths.items():
        records, sha = load(path), hashlib.sha256(path.read_bytes()).hexdigest()
        for fmt in FORMATS:
            built = [T.build_doc(r, fmt) for r in records]
            out[f"{fmt}@{tag}"] = {"sha": sha, "path": str(path),
                                   "docs": [{"key": d["key"], "embed": d["embed_text"], "bm25": d["bm25_text"]} for d in built]}
    return out


def doc_stats(docs: list[dict]) -> dict:
    chars = sorted(len(d["embed"]) for d in docs)
    q = lambda f: chars[min(len(chars) - 1, int(len(chars) * f))]  # noqa: E731
    texts = [d["embed"] for d in docs]
    return {"tables": len(docs), "chars median": q(.5), "chars p90": q(.9), "chars max": chars[-1],
            "region summary lines": sum(t.count("지역 단위 ") for t in texts),
            "long label lists": sum(len(LONG_LIST.findall(t)) for t in texts)}


def doc_vectors(model_name: str, fmt: str, sha: str, docs: list[dict], max_tokens: int, state: dict, batch: int) -> np.ndarray:
    """Cached like lab_common.doc_vectors, named by the format and the tables file so the four cells never share."""
    name = f"{model_name.split('/')[-1]}__{fmt}__{sha[:8]}-T{max_tokens}-v0text"
    npy, meta = CACHE / f"{name}.npy", CACHE / f"{name}.json"
    keys = [d["key"] for d in docs]
    if npy.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8"))["keys"] == keys:
        print(f"  cache {name}")
        return np.load(npy)
    model = model_for(model_name, max_tokens, state)
    started = time.perf_counter()
    vecs = np.asarray(encode(model, model_name, [d["embed"] for d in docs], "passage", batch), dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    meta.write_text(json.dumps({"name": name, "format": fmt, "source_sha256": sha, "model": model_name, "max_tokens": max_tokens,
                                "embed_seconds": round(time.perf_counter() - started, 1), "keys": keys}, ensure_ascii=False),
                    encoding="utf-8")
    print(f"  embedded {name} in {time.perf_counter() - started:.0f}s")
    return vecs


def question_sets() -> dict[str, dict]:
    """The two golden files as ft_eval reads them: ids, split, OR-groups (AND between groups), normalized questions."""
    rg = read_jsonl(ROUTING)
    rt_all = read_jsonl(RETRIEVAL)
    rt = [it for it in rt_all if it.get("answerable", True) and it.get("required")]
    sets = {
        "routing": {"items": rg, "ids": [it["id"] for it in rg], "split": [it["split"] for it in rg],
                    "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg], "qtype": ["-"] * len(rg)},
        "retrieval": {"items": rt, "ids": [it["qid"] for it in rt], "split": [it["split"] for it in rt],
                      "groups": [it["required"] for it in rt], "qtype": [it.get("qtype", "?") for it in rt]},
    }
    for s in sets.values():
        s["qn"] = [query_text(it["question"], "norm") for it in s["items"]]
        s["cluster"] = [g[0][0] for g in s["groups"]]
    return sets


def search(docs: list[dict], dvec: np.ndarray, qvec: np.ndarray, qn: list[str], tokenize) -> dict[str, list[list[str]]]:
    keys = [d["key"] for d in docs]
    bm = BM25Okapi([tokenize(d["bm25"]) for d in docs])
    bm25 = [[keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
            for sc in (np.asarray(bm.get_scores(tokenize(q))) for q in qn)]
    sc = dvec @ qvec.T
    dense = [top(sc[:, i], keys) for i in range(sc.shape[1])]
    return {"dense": dense, "bm25": bm25, "hybrid": [rrf(d, b) for d, b in zip(dense, bm25)]}


def fmt_row(cells: list) -> str:
    return "| " + " | ".join(f"{c:.3f}" if isinstance(c, float) else str(c) for c in cells) + " |"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="V0 hybrid search over table-text variants (plan 53)")
    p.add_argument("--old", type=Path, default=TABLES["old"], help="tables.jsonl before the rebuild (no levels)")
    p.add_argument("--new", type=Path, default=TABLES["new"], help="tables.jsonl after the rebuild (levels)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--docs-only", action="store_true", help="print text statistics per variant and stop")
    args = p.parse_args(argv)
    for path in (args.old, args.new):
        assert path.exists(), f"missing {path}"

    var = variants({"old": args.old, "new": args.new})
    print(f"{len(var)} variants; tables old {var['n2-L60@old']['sha'][:8]} new {var['n2-L60@new']['sha'][:8]}")
    stats = {name: doc_stats(v["docs"]) for name, v in var.items()}
    for name, st in stats.items():
        print(f"  {name:12s} " + ", ".join(f"{k} {v}" for k, v in st.items()))
    if args.docs_only:
        return

    sets = question_sets()
    tokenize = bm25_tokenize()
    state: dict = {}
    qvec = {name: question_vectors(args.model, s["qn"], Path(ROUTING if name == "routing" else RETRIEVAL).stem + "-ans", state, args.batch)
            for name, s in sets.items()}
    res: dict[str, dict[str, dict]] = defaultdict(dict)  # set -> variant -> {mode: ranks}
    lists: dict[str, dict[str, list]] = defaultdict(dict)
    for vname, v in var.items():
        fmt = vname.split("@")[0]
        dvec = doc_vectors(args.model, fmt, v["sha"], v["docs"], args.max_tokens, state, args.batch)
        for sname, s in sets.items():
            out = search(v["docs"], dvec, qvec[sname], s["qn"], tokenize)
            lists[sname][vname] = out["hybrid"]
            res[sname][vname] = {mode: [group_rank(g, t) for g, t in zip(s["groups"], ranked)] for mode, ranked in out.items()}
    state.clear()

    rng = np.random.default_rng(0)
    names = list(var)

    def block(sname: str, idx: list[int], title: str, ci: bool) -> list[str]:
        s = sets[sname]
        head = ["variant", *METRICS, "dense Hit@5", "BM25 Hit@5"] + (["dHit@5 vs baseline", "95% (question)", "95% (table)"] if ci else [])
        lines = [f"### {title} (n={len(idx)})", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        base = np.array([(res[sname][BASELINE]["hybrid"][i] or 999) <= 5 for i in idx], float)
        for vn in names:
            r = res[sname][vn]
            sm = summary([r["hybrid"][i] for i in idx])
            cells = [vn, *(sm[m] for m in METRICS), summary([r["dense"][i] for i in idx])["Hit@5"], summary([r["bm25"][i] for i in idx])["Hit@5"]]
            if ci:
                d = np.array([(r["hybrid"][i] or 999) <= 5 for i in idx], float) - base
                if vn == BASELINE:
                    cells += ["", "", ""]
                else:
                    lo, hi = boot(d, None, rng)
                    tlo, thi = boot(d, [s["cluster"][i] for i in idx], rng)
                    cells += [f"{d.mean():+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"[{tlo:+.3f}, {thi:+.3f}]"]
            lines.append(fmt_row(cells))
        return lines + [""]

    rs, ts = sets["routing"], sets["retrieval"]
    r_dev = [i for i, sp in enumerate(rs["split"]) if sp == "dev"]
    r_test = [i for i, sp in enumerate(rs["split"]) if sp == "test"]
    t_all = list(range(len(ts["ids"])))
    t_dev = [i for i, sp in enumerate(ts["split"]) if sp == "dev"]
    t_test = [i for i, sp in enumerate(ts["split"]) if sp == "test"]

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = [f"# Plan 53: V0 hybrid search over table-text variants ({args.model}, run {rid})", "",
             f"old tables: `{args.old}` sha {var['n2-L60@old']['sha'][:8]}; new tables: `{args.new}` sha {var['n2-L60@new']['sha'][:8]}. "
             f"Hybrid = dense top-{DEPTH} + BM25 top-{DEPTH} on the variant's own texts, RRF c=60. Baseline for the differences: {BASELINE}.", "",
             "## Texts", "", "| variant | tables | chars median | p90 | max | region summary lines | long label lists |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, st in stats.items():
        lines.append(f"| {name} | {st['tables']} | {st['chars median']} | {st['chars p90']} | {st['chars max']} | {st['region summary lines']} | {st['long label lists']} |")
    lines += ["", "## Routing golden (routing_v0)", ""]
    lines += block("routing", r_dev, "routing dev", ci=False)
    lines += block("routing", r_test, "routing test", ci=True)
    lines += ["## Retrieval golden (answerable rows; test seen before, exploratory)", ""]
    lines += block("retrieval", t_all, "retrieval all", ci=True)
    lines += block("retrieval", t_dev, "retrieval dev", ci=False)
    lines += block("retrieval", t_test, "retrieval test", ci=False)
    # Hit@5 by question type (retrieval all), hybrid
    qtypes = sorted(set(ts["qtype"]))
    lines += ["### retrieval all: hybrid Hit@5 by qtype", "", "| variant | " + " | ".join(f"{q} ({ts['qtype'].count(q)})" for q in qtypes) + " |",
              "|" + " --- |" * (len(qtypes) + 1)]
    for vn in names:
        cells = [vn]
        for q in qtypes:
            idx = [i for i, x in enumerate(ts["qtype"]) if x == q]
            cells.append(summary([res["retrieval"][vn]["hybrid"][i] for i in idx])["Hit@5"])
        lines.append(fmt_row(cells))
    lines += ["", "## Reading", "",
              "- n3@new vs n3@old: what the region-level summary (levels) adds; n3 vs n2 within one tables file: what the role/unit lines add.",
              "- An interval that contains 0 means no measurable effect at this sample size (routing test 104, retrieval 89 questions)."]
    report = "\n".join(lines) + "\n"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"text_v0_eval_{rid}_summary.md"
    summary_path.write_text(report, encoding="utf-8")
    ranks_path = OUT_DIR / f"text_v0_eval_{rid}_ranks.jsonl"
    with ranks_path.open("w", encoding="utf-8") as f:
        for sname, s in sets.items():
            for i, qid in enumerate(s["ids"]):
                f.write(json.dumps({"set": sname, "id": qid, "split": s["split"][i], "qtype": s["qtype"][i],
                                    **{vn: {"rank": res[sname][vn]["hybrid"][i], "top10": lists[sname][vn][i][:10]} for vn in names}},
                                   ensure_ascii=False) + "\n")
    print(report)
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
