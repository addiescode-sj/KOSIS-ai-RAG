"""T2-41 parity gate (1): dump the L1 ranking and scores (up to gate) per question, and compare two dumps.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/parity_l1.py dump --preset hybrid+cond --out data/parity/l1_s0_hybrid+cond.jsonl
    python scripts/parity_l1.py dump --preset staged [--questions <jsonl|txt>] --out data/parity/l1_s0_staged.jsonl
    python scripts/parity_l1.py compare data/parity/l1_s0_hybrid+cond.jsonl data/parity/l1_sN_hybrid+cond.jsonl

dump needs the real resources (tables.jsonl, the Chroma server, the models of the preset): it runs on the PC only.
The default questions are the L1 golden v2 questions, read by code (evals.schemas.load_items); --questions takes a
jsonl file (lines with "question", optional "qid") or a text file (one question per line, qid q0001, q0002, ...).
One line per question: {qid, question, keys, docs: [{key, score, rrf_score, rank_*, stage1_*}], removed, relaxed,
route, refusal_reason}; timings (*_seconds) and the trace are left out.

dump also writes `<out>.manifest.json`: preset, question source (path, sha256), expected (question count), qids_sha256,
written, completed (true only after the last line is flushed; a stopped run leaves false), and `config`, the effective
search configuration read off the built resources (Chroma collection, embedder, text format, tables.jsonl sha256,
active lists and stages, fusion weights incl. the hq weight, re-ranker checkpoints, cond_filter, pool).

compare refuses (exit 2) before looking at any row when a dump is empty or its manifest is missing, a run did not
complete, written != expected, the qid sets differ, the question files differ, the config fingerprints differ (the
differing keys are printed), or a dump's bytes are not the sha256 its manifest recorded. Then it prints `identical`
(exit 0) or the differing questions and fields (exit 1).
L1 only: route "guide" (S0..S4 name of the found path) is read as "found" (the S5 name) on both sides, so the S5
rename alone does not fail this gate; the app's own contract tests still compare the route values as they are.

Only build_preset_graph (the line that builds the graph) and config_fingerprint (where the resources keep their
settings) change in the later steps of the refactor; the fingerprint's keys and values must stay the same.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PRESETS = ("hybrid+cond", "staged")
DOC_FIELDS = ("key", "score", "rrf_score")
DOC_PREFIXES = ("rank_", "stage1_")
ROW_FIELDS = ("qid", "question", "keys", "docs", "removed", "relaxed", "route", "refusal_reason")
ROUTE_ALIASES = {"guide": "found"}  # L1 compare only: the S5 rename of the found path
REFUSE = 2  # exit code: the dumps cannot be compared (incomplete run, other questions or other configuration)


def default_questions_path() -> Path:
    """The L1 golden v2 file (evals/builders/golden_v2.py OUT_L1), next to evals.schemas.GOLDEN."""
    from evals.schemas import DATASETS

    return DATASETS / "retrieval_golden_v2.jsonl"


def read_questions(path: Path | None) -> list[tuple[str, str]]:
    """(qid, question) pairs: the golden v2 items by default, else a jsonl or a one-question-per-line text file."""
    if path is None:
        from evals.schemas import load_items

        return [(item.qid, item.question) for item in load_items(default_questions_path())]
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in lines]
        return [(str(row.get("qid") or f"q{n:04d}"), row["question"]) for n, row in enumerate(rows, 1)]
    return [(f"q{n:04d}", line.strip()) for n, line in enumerate(lines, 1)]


def build_preset_graph(preset: str):
    """(graph, resources) of a STAGED preset, built as the L1 runner's graph:<preset> builds it (S0 form)."""
    from kosis_chat.llm.model import MockClient

    from evals.runners.retrieval import _staged_settings
    from kosis_rag.graph import build_graph
    from kosis_rag.pipeline import load_resources

    resources = load_resources(_staged_settings(preset))
    return build_graph(resources, MockClient()), resources


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collection(retriever: Any) -> tuple[str | None, dict[str, Any]]:
    """(name, metadata) of a Chroma-backed retriever's collection; (None, {}) when it has none."""
    collection = getattr(retriever, "collection", None)
    if collection is None:
        return None, {}
    return getattr(collection, "name", None), dict(getattr(collection, "metadata", None) or {})


def config_fingerprint(preset: str, resources: Any) -> dict[str, Any]:
    """The effective search configuration, read off the built resources (never from .env).

    S0 form: resources = kosis_rag.pipeline.SearchResources (settings = SearchSettings). Later steps change only
    where these values are read from (options / spec); the keys and the values for the same setup stay equal.
    The list order and weights are the ones kosis_rag.graph.graph_nodes fuses with (dense 1.0, bm25 1.0, hq and
    hq_bm25 HQ_WEIGHT, each only when its retriever is loaded).
    """
    s = resources.settings
    dense_name, dense_meta = _collection(resources.dense)
    hq_name, _ = _collection(resources.hq)
    weights = {"dense": 1.0, "bm25": 1.0}
    if resources.hq is not None:
        weights["hq"] = float(s.hq_weight)
    if resources.hq_bm25 is not None:
        weights["hq_bm25"] = float(s.hq_weight)
    stage1 = resources.stage1_score is not None
    stage2 = resources.select is not None
    tables = Path(s.tables)
    return {
        "preset": preset,
        "tables_sha256": file_sha256(tables) if tables.is_file() else None,
        "collection": dense_name,
        "embed": dense_meta.get("embed_ckpt") or dense_meta.get("embed_model"),
        "collection_doc_format": dense_meta.get("doc_format"),
        "text_format": resources.text_format,
        "cond_filter": bool(s.cond_filter),
        "pool": int(s.pool),
        "lists": list(weights),
        "weights": weights,
        "stages": {"dense": True, "bm25": True, "hq": resources.hq is not None,
                   "hq_bm25": resources.hq_bm25 is not None, "rerank1": stage1, "rerank2": stage2},
        "hq_build": s.hq_build if resources.hq is not None else None,
        "hq_collection": hq_name,
        "rerank1_ckpt": s.rerank_ckpt if stage1 else None,
        "rerank1_mode": s.rerank_mode if stage1 else None,
        "rerank1_n": int(s.rerank_n) if stage1 else None,
        "rerank2": resources.stage2_name if stage2 else None,
        "rerank2_k": int(s.stage2_k) if stage2 else None,
    }


def manifest_path(out: Path) -> Path:
    return out.with_name(out.name + ".manifest.json")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest whole or not at all (tmp file, then rename)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                   newline="\n")
    tmp.replace(path)


def doc_row(metadata: dict[str, Any]) -> dict[str, Any]:
    """The ranking fields of one document: key, score, rrf_score, rank_*, stage1_* (no *_seconds)."""
    return {name: value for name, value in metadata.items()
            if (name in DOC_FIELDS or name.startswith(DOC_PREFIXES)) and not name.endswith("_seconds")}


def state_row(qid: str, question: str, state) -> dict[str, Any]:
    """One dump line from a final SearchState."""
    return {
        "qid": qid,
        "question": question,
        "keys": [d.metadata.get("key", d.id) for d in state.docs],
        "docs": [doc_row(d.metadata) for d in state.docs],
        "removed": state.removed,
        "relaxed": state.relaxed,
        "route": state.route,
        "refusal_reason": state.refusal_reason,
    }


def qids_sha256(qids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(qids)).encode("utf-8")).hexdigest()


def dump(preset: str, questions: Path | None, out: Path) -> int:
    from kosis_rag.graph import run_search

    source = questions if questions is not None else default_questions_path()
    pairs = read_questions(questions)
    qids = [qid for qid, _ in pairs]
    if len(set(qids)) != len(qids):
        print(f"refused: duplicate qids in {source}", file=sys.stderr)
        return REFUSE
    graph, resources = build_preset_graph(preset)
    out.parent.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(out)
    manifest = {"preset": preset, "questions": str(source), "questions_sha256": file_sha256(source),
                "expected": len(pairs), "qids_sha256": qids_sha256(qids), "written": 0, "completed": False,
                "out": out.name, "config": config_fingerprint(preset, resources)}
    write_manifest(mpath, manifest)  # completed false until the last line is on disk
    written = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for qid, question in pairs:
            row = state_row(qid, question, run_search(graph, question))
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            written += 1
        f.flush()
        os.fsync(f.fileno())
    write_manifest(mpath, {**manifest, "written": written, "completed": True, "out_sha256": file_sha256(out)})
    print(f"{preset}: {written} questions -> {out} (manifest {mpath.name})")
    return 0


def normalize_line(line: str) -> str:
    """The line with route "guide" read as "found" (ROUTE_ALIASES); other lines are returned as they are.

    Re-serialized the way dump writes (sort_keys, ensure_ascii=False): json round-trips ints, floats (repr), bools,
    None and key presence exactly, so only the route text changes.
    """
    row = json.loads(line)
    if row.get("route") in ROUTE_ALIASES:
        row["route"] = ROUTE_ALIASES[row["route"]]
        return json.dumps(row, sort_keys=True, ensure_ascii=False)
    return line


def read_lines(path: Path) -> dict[str, str]:
    """qid -> the raw JSON line, route normalized (dumps are written with sort_keys=True: equal rows, equal text)."""
    lines: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            line = normalize_line(line.strip())
            qid = json.loads(line)["qid"]
            if qid in lines:
                raise ValueError(f"{path}: duplicate qid {qid}")
            lines[qid] = line
    return lines


def read_dump(path: Path) -> dict[str, dict[str, Any]]:
    return {qid: json.loads(line) for qid, line in read_lines(path).items()}


def diff_rows(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> list[str]:
    """Human-readable differences between two dumps (qid sets, then field by field per qid)."""
    lines = [f"only in a: {qid}" for qid in a if qid not in b] + [f"only in b: {qid}" for qid in b if qid not in a]
    for qid in (q for q in a if q in b):
        ra, rb = a[qid], b[qid]
        for name in sorted(set(ra) | set(rb)):
            va, vb = ra.get(name), rb.get(name)
            if va == vb and type(va) is type(vb):
                continue
            if name == "docs" and isinstance(va, list) and isinstance(vb, list):
                if len(va) != len(vb):
                    lines.append(f"{qid} docs: {len(va)} vs {len(vb)} documents")
                for i, (da, db) in enumerate(zip(va, vb)):
                    for field in sorted(set(da) | set(db)):
                        if da.get(field) != db.get(field) or type(da.get(field)) is not type(db.get(field)):
                            lines.append(f"{qid} docs[{i}].{field}: {da.get(field)!r} vs {db.get(field)!r}")
            else:
                lines.append(f"{qid} {name}: {va!r} vs {vb!r}")
    return lines


def diff_config(ca: dict[str, Any], cb: dict[str, Any]) -> list[str]:
    return [f"config.{key}: {ca.get(key)!r} vs {cb.get(key)!r}"
            for key in sorted(set(ca) | set(cb)) if ca.get(key) != cb.get(key)]


def check_dumps(a: Path, b: Path, la: dict[str, str], lb: dict[str, str]) -> list[str]:
    """Why the two dumps cannot be compared ([] = they can): completeness, question set, configuration."""
    problems: list[str] = []
    manifests = {}
    for path, lines in ((a, la), (b, lb)):
        if not lines:
            problems.append(f"{path}: empty dump")
        mpath = manifest_path(path)
        if not mpath.is_file():
            problems.append(f"{path}: manifest {mpath.name} missing (dump it again with this script)")
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        manifests[path] = m
        if m.get("completed") is not True:
            problems.append(f"{path}: run not completed (manifest completed={m.get('completed')!r})")
        if not isinstance(m.get("expected"), int) or m.get("written") != m.get("expected"):
            problems.append(f"{path}: written {m.get('written')!r} != expected {m.get('expected')!r}")
        if len(lines) != m.get("written"):
            problems.append(f"{path}: {len(lines)} lines in the dump, manifest says written {m.get('written')!r}")
        elif m.get("qids_sha256") and qids_sha256(list(lines)) != m["qids_sha256"]:
            problems.append(f"{path}: the dump's qids are not the manifest's question set")
        if m.get("out_sha256") and file_sha256(path) != m["out_sha256"]:
            problems.append(f"{path}: the dump's bytes are not the ones its manifest recorded (changed after dump)")
    if set(la) != set(lb):
        only_a, only_b = sorted(set(la) - set(lb)), sorted(set(lb) - set(la))
        problems.append(f"qid sets differ: {len(only_a)} only in a {only_a[:5]}, {len(only_b)} only in b {only_b[:5]}")
    if len(manifests) == 2:
        ma, mb = manifests[a], manifests[b]
        if ma.get("questions_sha256") != mb.get("questions_sha256"):
            problems.append(f"question files differ: {ma.get('questions')} vs {mb.get('questions')} (sha256)")
        ca, cb = ma.get("config"), mb.get("config")
        if not isinstance(ca, dict) or not isinstance(cb, dict) or not ca or not cb:
            problems.append("config fingerprint missing in a manifest")
        else:
            problems += diff_config(ca, cb)
    return problems


def compare(a: Path, b: Path) -> int:
    for path in (a, b):
        if not path.is_file():
            print(f"refused: {path}: no such dump", file=sys.stderr)
            return REFUSE
    la, lb = read_lines(a), read_lines(b)
    problems = check_dumps(a, b, la, lb)
    if problems:
        print("refused: the dumps are not comparable", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return REFUSE
    if la == lb:  # raw text per qid: catches 3 vs 3.0, True vs 1 and None vs missing, which == on values misses
        print("identical")
        return 0
    lines = diff_rows(read_dump(a), read_dump(b))
    lines += [f"{qid}: text differs (type or key presence; values compare equal)"
              for qid in la if qid in lb and la[qid] != lb[qid] and not any(x.startswith(f"{qid} ") for x in lines)]
    print("\n".join(lines))
    print(f"{len(lines)} differences")
    return 1


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows cp949 console: never fail on a character outside the code page
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_dump = sub.add_parser("dump", help="run a preset over the questions and write one line per question")
    p_dump.add_argument("--preset", choices=PRESETS, required=True)
    p_dump.add_argument("--questions", type=Path, default=None,
                        help="jsonl (question, optional qid) or txt (one per line); default: L1 golden v2")
    p_dump.add_argument("--out", type=Path, required=True)
    p_cmp = sub.add_parser("compare", help="print identical, or the differing questions and fields (2: refused)")
    p_cmp.add_argument("a", type=Path)
    p_cmp.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    if args.command == "dump":
        return dump(args.preset, args.questions, args.out)
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
