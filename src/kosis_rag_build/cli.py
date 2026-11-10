"""The offline build CLI (T2-41 S6a): one entry point for every build step, each a graph of this package.

    python -m kosis_rag_build.cli tables   --data-dir <provided data> [--out data/tables.jsonl]
    python -m kosis_rag_build.cli sigungu  --data-dir <provided data> [--out data/sigungu.json]
    python -m kosis_rag_build.cli chroma   [--model kure|bge-m3-ft] [--ckpt ...] [--export x.npz | --import x.npz | --list]
    python -m kosis_rag_build.cli text     [--tables ...] [--tokens]          # table text formats: lengths and samples
    python -m kosis_rag_build.cli hq-build --model gpt-4o-mini [--limit 20] [--no-score | --rerank-ckpt ...]
    python -m kosis_rag_build.cli hq-index --build <build_id> [--ckpt ...] [--desc] [--dry-run]
    python -m kosis_rag_build.cli chunks   --data-dir <provided data> [--table KEY ...] [--registry-only]   # T2-43

(Windows cmd: `set PYTHONPATH=.;src;vendor/phase1;vendor/phase1/src`; bash: PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src.)

Settings come from the arguments, then from the `env` mapping the caller passes (§0.6: no .env is read). This
module reads no environment itself: `python scripts/kosis_build.py <command> ...` passes the shell environment
(`set X=...` first), `python -m kosis_rag_build.cli` uses arguments only. Names:
CHROMA_HOST CHROMA_PORT CHROMA_COLLECTION (chroma --list: the active build) EMBED_CKPT (chroma bge-m3-ft, hq-index)
TEXT_FORMAT (chroma bge-m3-ft) RERANK_CKPT (hq-build score filter) OPENAI_API_KEY (hq-build; never printed). They
become settings objects (ChromaConfig, IndexSpec, LLMConfig(provider="openai")) and the step's graph runs on them.
No file of src reads the environment (tests/test_layout.py; plans/T2-41.md 미결 3, 11/05 user decision).

Replaces the old module CLIs of the search package (metadata, shared.vocab, index, shared.tabletext, hq.build,
hq.index — the table in docs/migration.md S6a). hq-build prints token counts only: the cost is the evals side's (from build.json totals).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections.abc import Mapping
from pathlib import Path

from kosis_rag.llm.config import LLMConfig
from kosis_rag.schemas.resources import ChromaConfig, IndexSpec
from kosis_rag.shared import constants
from kosis_rag.shared import tabletext as text
from kosis_rag.shared.vocab import SIGUNGU
from kosis_rag_build import chroma  # stdlib + pydantic only; tables (pyarrow) is imported by its command
from kosis_rag_build.hq import constants as hq_constants

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"  # = kosis_rag_build.tables.DATA_DIR (not imported here: pyarrow loads only for `tables`)


# --- settings from the arguments and the shell environment ---

def _env(env: Mapping[str, str], name: str) -> str | None:
    """The variable's value, None when unset or empty."""
    return env.get(name) or None


def chroma_config(args: argparse.Namespace, env: Mapping[str, str]) -> ChromaConfig:
    """--chroma-host / --chroma-port, else CHROMA_HOST / CHROMA_PORT, else localhost:8001 (as the old CLIs read them
    from the environment or .env); collection = CHROMA_COLLECTION, the table build the search uses (chroma --list
    marks it). The collection a command writes is its own --collection argument, not this.
    SystemExit naming CHROMA_PORT when it is set but not a whole number (T2-41 S6b)."""
    port = args.chroma_port or _env(env, "CHROMA_PORT") or 8001
    try:
        port = int(port)
    except ValueError:
        raise SystemExit(f"CHROMA_PORT must be a port number, got {port!r}") from None
    return ChromaConfig(host=args.chroma_host or _env(env, "CHROMA_HOST") or "localhost", port=port,
                        collection=_env(env, "CHROMA_COLLECTION"))


def index_spec(args: argparse.Namespace, env: Mapping[str, str]) -> IndexSpec:
    """The index settings of chroma / hq-index / hq-build: tables, Chroma, embedder and re-ranker checkpoints, text
    format and device (arguments first, then EMBED_CKPT / RERANK_CKPT / TEXT_FORMAT)."""
    return IndexSpec(
        tables=getattr(args, "tables", None),
        chroma=chroma_config(args, env) if hasattr(args, "chroma_host") else ChromaConfig(),
        text_format=getattr(args, "text_format", None) or _env(env, "TEXT_FORMAT"),
        embed_ckpt=getattr(args, "ckpt", None) or _env(env, "EMBED_CKPT"),
        rerank_ckpt=getattr(args, "rerank_ckpt", None) or _env(env, "RERANK_CKPT"),
        device=getattr(args, "device", None),
    )


def hq_llm_config(args: argparse.Namespace, env: Mapping[str, str]) -> LLMConfig:
    """The question generator: OpenAI chat with the prompt's settings (temperature, max_tokens, seed) and
    OPENAI_API_KEY (None when unset: only cached tables succeed)."""
    return LLMConfig(provider="openai", api_key=_env(env, "OPENAI_API_KEY"), model=args.model,
                     temperature=hq_constants.TEMPERATURE, max_tokens=hq_constants.MAX_TOKENS, seed=hq_constants.SEED)


def _chroma_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chroma-host", default=None, help="Chroma server (default: CHROMA_HOST, else localhost)")
    p.add_argument("--chroma-port", type=int, default=None, help="default: CHROMA_PORT, else 8001")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m kosis_rag_build.cli",
                                 description="Offline DB preparation for kosis_rag (T2-41 S6a).")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tables", help="build data/tables.jsonl from the provided data (T2-01)")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=DATA / "tables.jsonl")

    p = sub.add_parser("sigungu", help="build data/sigungu.json (시군구 -> 시도) from the provided data")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=SIGUNGU)

    p = sub.add_parser("chroma", help="load the table metadata into ChromaDB (T2-24)")
    p.add_argument("--tables", type=Path, default=DATA / "tables.jsonl")
    p.add_argument("--model", choices=("kure", "bge-m3-ft"), default="kure",
                   help="kure: KURE-v1 over the T2-27 text (default); bge-m3-ft: fine-tuned bge-m3 (T2-06)")
    p.add_argument("--ckpt", default=None, help="bge-m3-ft: checkpoint .pt or run folder (default: EMBED_CKPT)")
    p.add_argument("--text-format", default=None,
                   help="bge-m3-ft: kosis_rag.shared.tabletext format (default: TEXT_FORMAT, else n2-L60)")
    p.add_argument("--collection", default=None,
                   help=f"default: kure {chroma.COLLECTION_PREFIX}<doc_format>__<hash of tables.jsonl and formats>; "
                        f"bge-m3-ft {chroma.FT_COLLECTION_PREFIX}<YYYYMMDDHHMM>")
    p.add_argument("--device", default=None, help="cpu, mps or cuda (default: auto)")
    p.add_argument("--list", action="store_true", help="list the builds on the server and which one is active")
    p.add_argument("--export", type=Path, default=None,
                   help="write the build to this .npz instead of Chroma (e.g. embed on a Colab GPU, T2-11)")
    p.add_argument("--import", dest="import_", type=Path, default=None,
                   help="load a build written by --export into Chroma (no embedding)")
    _chroma_args(p)

    p = sub.add_parser("chunks", help="region chunks of the very large tables: registry + Chroma collection (T2-43)")
    p.add_argument("--data-dir", type=Path, required=True, help="the provided data folder holding values/<key>.parquet")
    p.add_argument("--tables", type=Path, default=DATA / "tables.jsonl")
    p.add_argument("--table", action="append", default=None, help="table key; repeat it (default: the two largest tables)")
    p.add_argument("--out", type=Path, default=DATA / "chunks" / "regions.jsonl", help="the region registry file")
    p.add_argument("--registry-only", action="store_true", help="write the registry, do not embed or load")
    p.add_argument("--collection", default=None, help="default: kosis_chunks__<format>__<hash of the Parquet files>")
    p.add_argument("--device", default=None, help="cpu, mps or cuda (default: auto)")
    _chroma_args(p)

    p = sub.add_parser("text", help="table text formats: lengths and samples (kosis_rag.shared.tabletext)")
    p.add_argument("--tables", type=Path, default=DATA / "tables.jsonl")
    p.add_argument("--tokens", action="store_true", help="also count bge-m3 tokens (downloads the tokenizer)")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--out", type=Path, default=DATA / "text_formats_samples.md")  # data/ is gitignored

    p = sub.add_parser("hq-build", help="generate, check and filter hypothetical questions per table (T2-39)")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None, help="first N tables of tables.jsonl only (a sample build)")
    p.add_argument("--concurrency", type=int, default=hq_constants.CONCURRENCY)
    p.add_argument("--tables", type=Path, default=DATA / "tables.jsonl")
    p.add_argument("--rerank-ckpt", default=None, help="stage 1 cross-encoder (default: RERANK_CKPT)")
    p.add_argument("--no-score", action="store_true", help="rules only, no score filter (build id ...-f0)")
    p.add_argument("--frac", type=float, default=hq_constants.FILTER_FRAC, help="global share dropped by score")
    p.add_argument("--min-keep", type=int, default=hq_constants.MIN_KEEP, help="questions kept per table at least")
    p.add_argument("--text-format", default=text.DOC_FORMAT, help="table text format (n2-L60)")
    p.add_argument("--device", default=None, help="cpu or cuda for the cross-encoder (default: auto)")

    p = sub.add_parser("hq-index", help="load a hypothetical question build into Chroma (T2-39)")
    p.add_argument("--build", required=True, help="build id printed by hq-build")
    p.add_argument("--ckpt", default=None, help="fine-tuned bge-m3 .pt (default: EMBED_CKPT)")
    p.add_argument("--desc", action="store_true", help="index the descriptions instead (build id <id>-desc)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--collection", default=None, help="override the collection name")
    p.add_argument("--hq-root", type=Path, default=None, help="folder of the hq builds (default: data/hq)")
    p.add_argument("--dry-run", action="store_true",
                   help="print data_sha256, the collection name and the document count; no embedding, no Chroma")
    _chroma_args(p)
    return ap


# --- commands ---

def run_tables(args: argparse.Namespace) -> int:
    from kosis_rag_build.graph.index_build import build_tables_graph
    from kosis_rag_build.schemas import TablesBuildState

    started = time.perf_counter()
    state = TablesBuildState.model_validate(
        build_tables_graph().invoke(TablesBuildState(data_dir=args.data_dir, out=args.out)))
    print(f"{state.count} tables -> {args.out} ({time.perf_counter() - started:.0f}s)")
    return 0


def run_sigungu(args: argparse.Namespace) -> int:
    from kosis_rag_build.graph.index_build import build_sigungu_graph
    from kosis_rag_build.schemas import SigunguBuildState
    from kosis_rag_build.sigungu import SIGUNGU_SOURCE

    state = SigunguBuildState.model_validate(build_sigungu_graph().invoke(
        SigunguBuildState(parquet=args.data_dir / SIGUNGU_SOURCE, out=args.out)))
    print(f"wrote {args.out}: {len(state.mapping)} 시군구")
    return 0


def run_chroma(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    spec = index_spec(args, env)
    if args.list:
        from kosis_rag.adapters.chroma import chroma_client

        return chroma.list_builds(chroma_client(spec.chroma), spec.chroma.collection)
    if args.import_:
        from kosis_rag.adapters.chroma import chroma_client

        build = chroma.read_build(args.import_)
        count = chroma.load(chroma_client(spec.chroma), build["ids"], build["docs"], build["metas"], build["vectors"],
                            build["name"], build["record"])
        print(f"collection {build['name']}: {count} documents")
        print(f"to search it, set CHROMA_COLLECTION={build['name']}")
        return 0 if count == len(build["ids"]) else 1

    from kosis_rag_build.graph.index_build import build_chroma_graph
    from kosis_rag_build.schemas import IndexBuildState

    state = IndexBuildState(tables=spec.tables, model=args.model, ckpt=spec.embed_ckpt, text_format=spec.text_format,
                            collection=args.collection, device=spec.device, export=args.export, chroma=spec.chroma)
    state = IndexBuildState.model_validate(build_chroma_graph().invoke(state))
    if args.export:
        print(f"wrote {args.export}; load it with: python -m kosis_rag_build.cli chroma --import {args.export.name}")
        return 0
    print(f"collection {state.name}: {state.count} documents")
    print(f"to search it, set CHROMA_COLLECTION={state.name} (the previous builds stay; switch back the same way)")
    return 0 if state.count == len(state.ids) else 1


def run_chunks(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    """Registry (one group-by per table) -> documents -> KURE-v1 -> its own collection. Values are not loaded."""
    from kosis_rag.adapters import files
    from kosis_rag_build import chunks as ch
    from kosis_rag_build.schemas import index_version_now

    keys = args.table or list(ch.CHUNK_TABLES)
    infos = {r.info.key: r.info for r in files.read_records(args.tables) if r.info.key in keys}
    if missing := [k for k in keys if k not in infos]:
        raise SystemExit(f"not in {args.tables}: {', '.join(missing)}")
    started = time.perf_counter()
    registry, hashes = [], []
    for key in keys:
        parquet = args.data_dir / infos[key].parquet
        part = ch.region_registry(parquet, infos[key])
        if (n := sum(c.rows for c in part)) != infos[key].rows:
            raise SystemExit(f"{key}: chunks cover {n} rows, tables.jsonl says {infos[key].rows}")
        hashes.append(ch.file_sha256(parquet))
        registry += part
        print(f"{key}: {len(part)} regions, {n} rows")
    ch.write_registry(args.out, registry)
    print(f"wrote {args.out}: {len(registry)} chunks ({time.perf_counter() - started:.0f}s)")
    if args.registry_only:
        return 0

    from kosis_rag.adapters.chroma import chroma_client
    from kosis_rag.adapters.models import kure_embedder

    docs = [ch.to_document(c, infos[c.table_key]) for c in registry]
    metas = [ch.to_metadata(c, infos[c.table_key]) for c in registry]
    embed, count_tokens, max_tokens = kure_embedder(args.device)
    longest = max(count_tokens(d) for d in docs)
    if longest > max_tokens:
        raise SystemExit(f"a chunk has {longest} tokens, over the model limit {max_tokens}")
    started = time.perf_counter()
    vectors = [v for i in range(0, len(docs), 256) for v in embed(docs[i:i + 256])]
    print(f"embedded {len(docs)} chunks ({time.perf_counter() - started:.0f}s), max {longest} tokens")
    data_sha256 = "|".join(hashes)
    name = args.collection or ch.collection_name(data_sha256)
    record = ch.chunk_record(index_version_now(), hashlib.sha256(data_sha256.encode()).hexdigest(), registry,
                             len(vectors[0]), longest)
    count = chroma.load(chroma_client(chroma_config(args, env)), [c.id for c in registry], docs, metas, vectors,
                        name, record)
    print(f"collection {name}: {count} chunks (the table collection is unchanged)")
    return 0 if count == len(registry) else 1


def run_text(args: argparse.Namespace) -> int:
    """Length per format (characters, and tokens with --tokens) and before/after samples for a few tables."""
    import random

    from kosis_rag.adapters.files import read_records

    records = read_records(args.tables)
    tok = None
    if args.tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    docs = {fmt: [text.build_doc(r, fmt)["embed_text"] for r in records] for fmt in text.FORMATS}

    def pct(xs: list[int], q: float) -> int:
        return sorted(xs)[min(len(xs) - 1, int(len(xs) * q))]

    print(f"{len(records)} tables")
    for fmt, texts in docs.items():
        chars = [len(t) for t in texts]
        line = f"  {fmt}: chars median {pct(chars, .5)}, p90 {pct(chars, .9)}, max {max(chars)}"
        if tok is not None:
            n = [len(tok(t)["input_ids"]) for t in texts]
            line += (f" | tokens median {pct(n, .5)}, p90 {pct(n, .9)}, max {max(n)}, "
                     f"> 480 (re-ranker pair 512 minus a question) {sum(x > 480 for x in n)}, > 1024 {sum(x > 1024 for x in n)}")
        print(line)
    random.seed(0)
    picks = random.sample(range(len(records)), min(args.samples, len(records)))
    with args.out.open("w", encoding="utf-8") as f:
        for i in picks:
            f.write(f"## {records[i].info.key}\n원문 표 이름: {records[i].info.table_name}\n\n")
            for fmt in text.FORMATS:
                f.write(f"### {fmt}\n```\n{docs[fmt][i]}\n```\n\n")
    print(f"samples -> {args.out}")
    return 0


def tokens_line(totals: dict) -> str:
    """Token counts of a build (this run's calls and all replies); the cost is computed on the evals side from
    build.json totals (T2-41 S6a: was hq.build.cost_line over evals/pricing.json)."""
    return (f"tokens this run: input {totals['new_input_tokens']:,} / output {totals['new_output_tokens']:,}; "
            f"all replies: input {totals['input_tokens']:,} / output {totals['output_tokens']:,} "
            "(cost: evals, from build.json totals)")


def run_hq_build(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    from kosis_rag.adapters.files import read_records
    from kosis_rag.llm.openai import OpenAIClient
    from kosis_rag_build.graph import hq_build
    from kosis_rag_build.hq import nodes as hq_nodes
    from kosis_rag_build.hq import prompt as hq_prompt

    spec, llm = index_spec(args, env), hq_llm_config(args, env)
    if not llm.api_key:
        print("warning: OPENAI_API_KEY is empty: only tables already in the cache will succeed")
    records = read_records(spec.tables)
    sha = hq_nodes.tables_sha(spec.tables)
    scorer = None
    if not args.no_score:
        if not spec.rerank_ckpt:
            raise SystemExit("set --rerank-ckpt (or RERANK_CKPT), or pass --no-score")
        scorer = hq_nodes.pair_scorer(spec.rerank_ckpt, records, args.text_format, spec.device)
    client = OpenAIClient(llm)
    n = min(args.limit or len(records), len(records))
    print(f"prompt {hq_constants.PROMPT_VERSION} ({hq_prompt.prompt_hash()[:8]}), model {args.model}, "
          f"{n} tables, concurrency {args.concurrency}, cache {hq_nodes.CACHE_DIR / hq_prompt.prompt_hash()[:8]}")
    state = hq_build.run_build(records, client, model=args.model, tables_sha256=sha, limit=args.limit,
                               frac=args.frac, min_keep=args.min_keep, scorer=scorer, concurrency=args.concurrency,
                               fmt=args.text_format)
    t = state.totals
    print(f"build {state.build_id} -> {state.out_dir}")
    print(f"tables {t['tables']} (failed {t['failed']}), questions generated {t['generated']}, "
          f"rule-dropped {t['rule_dropped']}, score-dropped {t['score_dropped']}, kept {t['kept']}")
    print(f"replies {t['replies']} (API calls this run {t['calls']}, cached {t['cached']}), API {t['api_seconds']}s")
    print(tokens_line(t))
    print(f"next: python -m kosis_rag_build.cli hq-index --build {state.build_id} --ckpt <EMBED_CKPT>, "
          f"then HQ_BUILD={state.build_id}")
    return 0 if t["failed"] == 0 else 1


def run_hq_index(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    from kosis_rag_build.graph.index_build import build_questions_graph
    from kosis_rag_build.hq import questions_index
    from kosis_rag_build.schemas import QuestionsIndexState

    spec = index_spec(args, env)
    build = args.build + (constants.DESC_SUFFIX if args.desc and not args.build.endswith(constants.DESC_SUFFIX) else "")
    state = QuestionsIndexState(build=build, ckpt=spec.embed_ckpt, device=spec.device, collection=args.collection,
                                hq_root=args.hq_root, chroma=spec.chroma, dry_run=args.dry_run)
    state = QuestionsIndexState.model_validate(build_questions_graph().invoke(state))
    questions_index.summary(state)
    return 0 if state.dry_run or state.count == len(state.rows) else 1


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Parse argv and run the command. env = the settings fallback (scripts/kosis_build.py passes os.environ;
    None = no fallback, arguments and defaults only)."""
    args = parser().parse_args(argv)
    env = {} if env is None else env
    if args.command == "tables":
        return run_tables(args)
    if args.command == "sigungu":
        return run_sigungu(args)
    if args.command == "chroma":
        return run_chroma(args, env)
    if args.command == "chunks":
        return run_chunks(args, env)
    if args.command == "text":
        return run_text(args)
    if args.command == "hq-build":
        return run_hq_build(args, env)
    return run_hq_index(args, env)


if __name__ == "__main__":
    raise SystemExit(main())
