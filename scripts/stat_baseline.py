"""Phase 1 baseline through the public API: golden questions -> summarize_stat(question, table_id) -> a graded run.

    python scripts/stat_baseline.py --stage p2-baseline --backend hcx --model HCX-DASH-002 --interval 5
    python scripts/phase1.py grade_run --run eval/results/<run>

Each golden question goes to `summarize_stat` with its table's tables.csv key (Path(item.table).stem), the key
stage 2 search will hand over. The run folder (manifest.json, golden.jsonl, results.jsonl) is the one
eval.grade_run reads, so the result compares question by question with a Phase 1 `run_eval --pipeline stat` run.

Differences from run_eval, on purpose: no golden scope (summarize_stat has none; it only matters for the
compact-v2 fallback), no route metadata (grade_run shows routes as "-"), no trace recording.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "vendor" / "phase1"


def _setup() -> None:
    os.environ.setdefault("KOSIS_WORKSPACE", str(ROOT))  # docs/ISSUES.md P2-07
    for path in (PHASE1 / "src", PHASE1):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _copy_golden(src: Path, dst: Path, keep: set[str] | None) -> None:
    """The golden lines the run asked, as one .jsonl (a byte copy when it is one file and every question ran)."""
    files = sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
    if keep is None and len(files) == 1:
        shutil.copyfile(files[0], dst)
        return
    lines = [line for f in files for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [line for line in lines if keep is None or json.loads(line)["id"] in keep]
    dst.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def run(golden: Path, stage: str, settings, interval: float = 0.0, limit: int | None = None,
        out_root: Path | None = None, client=None) -> Path:
    """Answer every golden question once through summarize_stat; returns the run folder."""
    from eval.golden import load_items
    from kosis_chat import api
    from kosis_chat.config import resolve_path
    from kosis_chat.pipeline import VERSION

    if interval < 0:
        raise ValueError("interval cannot be negative")
    version, items, golden_sha = load_items(golden)  # schema, table files and their sha256 are checked here
    items = items[:limit] if limit else items
    for item in items:  # the key route must reach the very file the golden answer was written against
        if api.table_path(Path(item.table).stem).resolve() != item.path.resolve():
            raise LookupError(f"{item.id}: tables.csv key {Path(item.table).stem} is not {item.table}")

    started = dt.datetime.now().astimezone()
    run_id = f"{started:%Y%m%d-%H%M%S-%f}-{stage}"
    run_dir = (out_root or resolve_path("eval/results")) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _copy_golden(golden, run_dir / "golden.jsonl", {i.id for i in items} if limit else None)
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "started_at": started.isoformat(timespec="seconds"),
        "git_commit": "unknown",  # no Git in the OneDrive checkout
        "entry": "kosis_chat.api.summarize_stat",
        "pipeline": "stat",
        "golden_version": version,
        "golden_sha256": golden_sha,
        "data_sha256": {i.table: i.table_sha256 for i in items},
        "prompt_version": VERSION,  # the C pipeline's own version, as run_eval records it
        "interval_seconds": interval,
        "serialization": settings.serialization,
        "scope_mode": "auto",
        "model": settings.model_params(),
        "runs": 1,
        "questions": len(items),
        "pretest": settings.backend == "mock",
        "status": "running",
    }

    def write_manifest() -> None:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_manifest()
    t_run, waited, failed = time.perf_counter(), 0.0, 0
    try:
        with (run_dir / "results.jsonl").open("w", encoding="utf-8") as out:
            for n, item in enumerate(items, start=1):
                if interval and n > 1:
                    time.sleep(interval)  # stay under per-minute rate limits (HCX 429), as run_eval --interval
                    waited += interval
                key = Path(item.table).stem
                t0 = time.perf_counter()
                try:
                    stat = api.summarize_stat(item.question, key, settings=settings, client=client, retry_interval=interval)
                    error = None
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # one question's failure is that question's run_failure; the rest still run
                    stat, error = None, f"{type(exc).__name__}: {exc}"
                latency = time.perf_counter() - t0
                failed += stat is None
                row = {
                    "question_id": item.id,
                    "kind": item.kind,
                    "run_index": 0,
                    "question": item.question,
                    "table_key": key,
                    "call_status": "ok" if stat else "error",
                    "call_error": error,
                    "latency_seconds": round(latency, 3),
                    "decision_metadata": {},  # summarize_stat returns the StatAnswer only
                    "stat_answer": stat.model_dump(mode="json") if stat else None,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                outcome = stat.outcome if stat else error[:120]
                print(f"[{n}/{len(items)}] {item.id} {latency:.1f}s {outcome}", flush=True)
        manifest["status"] = "finished"
    except BaseException as exc:
        manifest["status"] = "stopped"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        manifest["timing"] = {"total_s": round(time.perf_counter() - t_run, 2), "interval_wait_s": round(waited, 2)}
        manifest["failed_calls"] = failed
        write_manifest()
    return run_dir


def main(argv: list[str] | None = None) -> int:
    _setup()
    from kosis_chat.config import load_env, load_settings, resolve_path

    parser = argparse.ArgumentParser(prog="python scripts/stat_baseline.py", description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, help="label for the run folder, e.g. p2-baseline")
    parser.add_argument("--golden", default="eval/golden/golden.jsonl", help="golden .jsonl file or a directory of them")
    parser.add_argument("--backend", default=None, help="hcx | mock (default: configs/model.yaml)")
    parser.add_argument("--model", default=None, help="model name (default: configs/model.yaml)")
    parser.add_argument("--serialization", default="wide", choices=("long", "wide"),
                        help="table text for the compact-v2 fallback (default wide, as the Phase 1 baseline run)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between model calls (default 1)")
    parser.add_argument("--limit", type=int, default=None, help="answer only the first N questions")
    args = parser.parse_args(argv)
    load_env()  # HCX_API_KEY from <workspace>/.env; values are never printed
    settings = load_settings(backend=args.backend, model_name=args.model, serialization=args.serialization)
    run_dir = run(resolve_path(args.golden), args.stage, settings, interval=args.interval, limit=args.limit)
    print(f"run folder: {run_dir}")
    print(f"next: python scripts\\phase1.py grade_run --run \"{run_dir}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
