"""T2-41 parity gate (3): hash an hq (reverse HyDE) build folder, and compare two hash files.

    python scripts/parity_hq.py hash data/hq/hq-cb0b3c5b-a3c88540-gpt-4o-mini-f0 --out data/parity/hq_f0_s0.sha256
    python scripts/parity_hq.py compare data/parity/hq_f30k2n20_pc.sha256 data/parity/hq_f30k2n20_s0.sha256
    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/parity_hq.py replay \
        data/hq/hq-cb0b3c5b-a3c88540-gpt-4o-mini-f30-k2-n20 --out-dir data/parity/replay_f30k2n20_s0 \
        --rerank-ckpt <RERANK_CKPT> --hash-out data/parity/hq_f30k2n20_replay_s0.sha256

A hash file has one `<value>  <name>` line per entry, sorted by name:
- `questions.jsonl descriptions.jsonl failed.jsonl report.md build.json`: sha256 of each file as written ("missing"
  when absent);
- `build.json~`: sha256 of build.json with totals.{calls,cached,new_input_tokens,new_output_tokens,api_seconds}
  removed (what a rebuild from the cache changes), re-serialized with sorted keys;
- `report.md~`: sha256 of report.md without its "- 응답 ..." line (the same run-dependent totals, in words);
- `prompt_hash8`: build.json's prompt_hash[:8]; `model`: build.json's model; `calls`: totals.calls of that run;
- `cache_dir`: "present" or "missing" for data/hq/cache/<prompt_hash8>/<model slug>/ next to the build folder.

hash refuses (exit 2) when a file of FILES is missing. compare refuses (exit 2) when either hash file lacks a
REQUIRED entry or has "missing" for one of FILES / build.json~ / report.md~ / prompt_hash8 / model; then it prints
`identical` (exit 0) when every gate entry agrees, else the differing entries (exit 1). The raw build.json /
report.md hashes and `calls` are run-dependent: they are shown as notes, never fail the comparison.

replay rebuilds a baseline build folder from the reply cache only, into a NEW folder: --out-dir must not exist or be
empty and must not be (or contain, or sit inside) the baseline folder; the build lands in <out-dir>/<build_id>, so
the baseline is never overwritten. The settings come from the baseline's build.json (model, filter_frac, min_keep,
limit); prompt_hash, tables.jsonl sha256 and the build id must match the baseline before anything runs. The LLM
client is NoCallClient: any call (= a cache miss) prints the miss and raises CacheMiss, a BaseException, so neither
the table graph's `except Exception` nor the build graph's catches it: the run stops, nothing is written, exit 3.
A baseline with an API failure in failed.jsonl therefore cannot be replayed (that table was never cached).
The cache is only read (a reply is written to it only after a call, which never happens). After the run: exit 3 if
the client was called at all or totals.calls != 0, exit 2 if a FILES entry is missing; otherwise the hash file of
the replayed folder (cache_dir judged against the cache used) is written to --hash-out (default
<out-dir>/<build_id>.sha256) and compared with the baseline folder's hashes, `identical` = exit 0.
Filtered builds (filter_frac > 0) need the stage 1 cross-encoder: --rerank-ckpt (this script reads no .env).

hash and compare read only the build folder; no import of kosis_rag (they run without the search dependencies).
replay imports kosis_rag.hq (needs PYTHONPATH=.;src;vendor/phase1;vendor/phase1/src and the build dependencies).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

FILES = ("questions.jsonl", "descriptions.jsonl", "failed.jsonl", "report.md", "build.json")
RUN_TOTALS = ("calls", "cached", "new_input_tokens", "new_output_tokens", "api_seconds")
REPORT_RUN_LINE = "- 응답 "  # report.md line with the run-dependent totals (kosis_rag.hq.graph.report)
# Entries that differ between a build from the API and the same build from the cache: notes only
RUN_DEPENDENT = ("build.json", "report.md", "calls")
MISSING = "missing"
# Entries every hash file must have; the ones in MUST_HAVE_VALUE must also not be "missing"
REQUIRED = (*FILES, "build.json~", "report.md~", "prompt_hash8", "model", "calls", "cache_dir")
MUST_HAVE_VALUE = (*FILES, "build.json~", "report.md~", "prompt_hash8", "model")
REFUSE = 2  # exit code: incomplete input, nothing compared
CACHE_MISS = 3  # exit code: replay needed an LLM call


class Refused(Exception):
    """The input is incomplete or unsafe; the message says why (exit REFUSE)."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_slug(model: str) -> str:
    """As kosis_rag.hq.graph.model_slug (copied so this script imports no search code)."""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:40].strip("-") or "model"


def missing_files(build_dir: Path) -> list[str]:
    return [name for name in FILES if not (build_dir / name).is_file()]


def build_hashes(build_dir: Path, cache_root: Path | None = None) -> dict[str, str]:
    """name -> value for one build folder (see the module docstring).

    cache_root = the cache the build was made from (default: data/hq/cache next to the build folder).
    Raises Refused when the folder or one of FILES is missing.
    """
    if not build_dir.is_dir():
        raise Refused(f"not a build folder: {build_dir}")
    absent = missing_files(build_dir)
    if absent:
        raise Refused(f"{build_dir}: required files missing: {' '.join(absent)}")
    out = {}
    for name in FILES:
        path = build_dir / name
        out[name] = sha256(path.read_bytes()) if path.is_file() else MISSING
    manifest_path = build_dir / "build.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    totals = dict(manifest.get("totals") or {})
    stable = {**manifest, "totals": {k: v for k, v in totals.items() if k not in RUN_TOTALS}}
    out["build.json~"] = (sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False).encode("utf-8"))
                          if manifest else MISSING)
    report_path = build_dir / "report.md"
    if report_path.is_file():
        lines = report_path.read_text(encoding="utf-8").splitlines()
        out["report.md~"] = sha256("\n".join(l for l in lines if not l.startswith(REPORT_RUN_LINE)).encode("utf-8"))
    else:
        out["report.md~"] = MISSING
    prompt8 = str(manifest.get("prompt_hash") or "")[:8] or MISSING
    model = str(manifest.get("model") or "") or MISSING
    out["prompt_hash8"] = prompt8
    out["model"] = model
    out["calls"] = str(totals["calls"]) if "calls" in totals else MISSING
    cache = (cache_root if cache_root is not None else build_dir.parent / "cache") / prompt8 / model_slug(model)
    out["cache_dir"] = "present" if prompt8 != MISSING and model != MISSING and cache.is_dir() else "missing"
    return out


def write_hashes(hashes: dict[str, str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for name in sorted(hashes):
            f.write(f"{hashes[name]}  {name}\n")


def read_hashes(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value, _, name = line.partition("  ")
            out[name.strip()] = value.strip()
    return out


def lacking(hashes: dict[str, str]) -> list[str]:
    """REQUIRED entries a hash file does not have (or has as "missing" where a value is required)."""
    return [name for name in REQUIRED
            if name not in hashes or (name in MUST_HAVE_VALUE and hashes[name] == MISSING)]


def compare(a: Path, b: Path) -> int:
    for path in (a, b):
        if not path.is_file():
            print(f"refused: {path}: no such hash file", file=sys.stderr)
            return REFUSE
    return compare_hashes(read_hashes(a), read_hashes(b), str(a), str(b))


def compare_hashes(ha: dict[str, str], hb: dict[str, str], name_a: str = "a", name_b: str = "b") -> int:
    refused = [f"{label}: lacks {' '.join(lacking(h))}" for label, h in ((name_a, ha), (name_b, hb)) if lacking(h)]
    if refused:
        print("refused: incomplete hash file", file=sys.stderr)
        for line in refused:
            print(f"  {line}", file=sys.stderr)
        return REFUSE
    gate, notes = [], []
    for name in sorted(set(ha) | set(hb)):
        va, vb = ha.get(name, MISSING), hb.get(name, MISSING)
        if va != vb:
            (notes if name in RUN_DEPENDENT else gate).append(f"{name}: {va} vs {vb}")
    for line in notes:
        print(f"note (run-dependent) {line}")
    if gate:
        print("\n".join(gate))
        print(f"{len(gate)} differences")
        return 1
    print("identical")
    return 0


class CacheMiss(BaseException):
    """Replay needed an LLM call. A BaseException on purpose: hq.graph's generate node and the build graph's
    per-table guard catch Exception (and record a failed table); this one passes both and stops the run."""


class NoCallClient:
    """The replay's LLM client: every call is a cache miss and stops the run (CacheMiss); `calls` counts them."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system: str, user: str, history: Any = None) -> Any:
        self.calls += 1
        first = " ".join(user.split())[:120]
        print(f"cache miss: the replay would call the LLM (call {self.calls}; user message {first!r})",
              file=sys.stderr, flush=True)
        raise CacheMiss(f"cache miss (call {self.calls})")


def _inside(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def check_out_dir(out_dir: Path, baseline: Path) -> None:
    """Refused unless out_dir is a new (absent or empty) folder apart from the baseline build folder."""
    out, base = out_dir.resolve(), baseline.resolve()
    if out == base or _inside(out, base) or _inside(base, out):
        raise Refused(f"--out-dir {out_dir} is, contains or sits inside the baseline {baseline}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refused(f"--out-dir {out_dir} exists and is not an empty folder")


def replay(baseline: Path, out_dir: Path, *, tables: Path, cache_root: Path | None, rerank_ckpt: str | None,
           text_format: str | None, device: str | None, concurrency: int, hash_out: Path | None) -> int:
    """Rebuild `baseline` from the cache into out_dir/<build_id> with NoCallClient (see the module docstring)."""
    if not baseline.is_dir():
        raise Refused(f"not a build folder: {baseline}")
    absent = missing_files(baseline)
    if absent:
        raise Refused(f"baseline {baseline}: required files missing: {' '.join(absent)}")
    check_out_dir(out_dir, baseline)
    manifest = json.loads((baseline / "build.json").read_text(encoding="utf-8"))
    cache_root = cache_root if cache_root is not None else baseline.parent / "cache"

    from kosis_rag import text
    from kosis_rag.hq import graph as hq_graph
    from kosis_rag.hq import prompt as hq_prompt
    from kosis_rag.pipeline import read_records

    model = manifest["model"]
    frac, min_keep, limit = float(manifest["filter_frac"]), int(manifest["min_keep"]), manifest.get("limit")
    prompt_hash = hq_prompt.prompt_hash()
    if prompt_hash != manifest.get("prompt_hash"):
        raise Refused(f"prompt_hash {prompt_hash[:8]} != baseline {str(manifest.get('prompt_hash'))[:8]}: every "
                      "table would be a cache miss")
    sha = hq_graph.tables_sha(tables)
    if sha != manifest.get("tables_sha256"):
        raise Refused(f"{tables}: sha256 {sha[:8]} != baseline tables_sha256 {str(manifest.get('tables_sha256'))[:8]}")
    bid = hq_graph.build_id(prompt_hash, sha, model, frac, min_keep, limit)
    if bid != baseline.name or bid != manifest.get("build_id"):
        raise Refused(f"the replay's build id {bid} is not the baseline's {baseline.name}")
    cache = cache_root / prompt_hash[:8] / hq_graph.model_slug(model)
    if not cache.is_dir():
        raise Refused(f"no reply cache at {cache}")
    fmt = text_format or text.DOC_FORMAT
    records = read_records(tables)
    scorer = None
    if frac > 0:
        if not rerank_ckpt:
            raise Refused(f"the baseline is filtered (filter_frac {frac}): pass --rerank-ckpt <stage 1 checkpoint>")
        from kosis_rag.hq.build import pair_scorer

        scorer = pair_scorer(rerank_ckpt, records, fmt, device)
    client = NoCallClient()
    print(f"replay {bid}: model {model}, limit {limit}, frac {frac}, cache {cache} -> {out_dir / bid}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        state = hq_graph.run_build(records, client, model=model, tables_sha256=sha, limit=limit, frac=frac,
                                   min_keep=min_keep, scorer=scorer, cache_dir=cache_root, out_root=out_dir,
                                   concurrency=concurrency, fmt=fmt)
    except CacheMiss as exc:
        print(f"replay failed: {exc}; nothing was written to {out_dir}", file=sys.stderr)
        return CACHE_MISS
    calls = state.totals.get("calls")
    if client.calls or calls != 0:  # belt and braces: CacheMiss should already have stopped the run
        print(f"replay failed: client called {client.calls} times, totals.calls {calls}", file=sys.stderr)
        return CACHE_MISS
    built = Path(state.out_dir)
    if built.resolve() != (out_dir / bid).resolve():
        raise Refused(f"the replay wrote {built}, expected {out_dir / bid}")
    replayed = build_hashes(built, cache_root)
    target = hash_out or out_dir / f"{bid}.sha256"
    write_hashes(replayed, target)
    print(f"replayed {bid}: calls 0, cached {state.totals.get('cached')}, failed {state.totals.get('failed')}, "
          f"kept {state.totals.get('kept')} -> {target}")
    print(f"compare with the baseline {baseline}:")
    return compare_hashes(build_hashes(baseline, cache_root), replayed, "baseline", "replay")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows cp949 console: never fail on a character outside the code page
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_hash = sub.add_parser("hash", help="write the hash file of one build folder (2: a required file is missing)")
    p_hash.add_argument("build_dir", type=Path)
    p_hash.add_argument("--out", type=Path, required=True)
    p_hash.add_argument("--cache-root", type=Path, default=None,
                        help="reply cache root for the cache_dir entry (default: cache next to the build folder)")
    p_cmp = sub.add_parser("compare", help="print identical, or the differing entries (2: incomplete hash file)")
    p_cmp.add_argument("a", type=Path)
    p_cmp.add_argument("b", type=Path)
    p_rep = sub.add_parser("replay", help="rebuild a build folder from the cache only, into a new folder")
    p_rep.add_argument("baseline", type=Path, help="the build folder to replay (data/hq/<build_id>)")
    p_rep.add_argument("--out-dir", type=Path, required=True, help="new, empty folder; the build goes to <it>/<id>")
    p_rep.add_argument("--tables", type=Path, default=Path("data") / "tables.jsonl")
    p_rep.add_argument("--cache-root", type=Path, default=None, help="default: cache next to the baseline folder")
    p_rep.add_argument("--rerank-ckpt", default=None, help="stage 1 cross-encoder (filtered builds only)")
    p_rep.add_argument("--text-format", default=None, help="table text format (default: kosis_rag.text.DOC_FORMAT)")
    p_rep.add_argument("--device", default=None)
    p_rep.add_argument("--concurrency", type=int, default=8)
    p_rep.add_argument("--hash-out", type=Path, default=None, help="default: <out-dir>/<build_id>.sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "hash":
            hashes = build_hashes(args.build_dir, args.cache_root)
            write_hashes(hashes, args.out)
            print(f"{args.build_dir.name}: prompt {hashes['prompt_hash8']}, calls {hashes['calls']}, "
                  f"cache {hashes['cache_dir']} -> {args.out}")
            return 0
        if args.command == "replay":
            return replay(args.baseline, args.out_dir, tables=args.tables, cache_root=args.cache_root,
                          rerank_ckpt=args.rerank_ckpt, text_format=args.text_format, device=args.device,
                          concurrency=args.concurrency, hash_out=args.hash_out)
        return compare(args.a, args.b)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return REFUSE


if __name__ == "__main__":
    sys.exit(main())
