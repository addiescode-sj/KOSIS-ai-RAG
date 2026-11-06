"""The nodes of the offline question build (T2-39; T2-41 S6a: the node bodies and helpers of the old hq/graph.py,
the pair scorer of the old hq/build.py). The graphs over them are graph/hq_build.py; report.md is hq/report.py.

Per table (table_nodes, state HqState): load -> prompt -> generate -> validate -> rule_check -> [score].
Over all tables (build_nodes, state HqBuildState): generate_all -> filter_all -> write.

Fan-out: generate_all invokes the compiled table graph from a ThreadPoolExecutor (default 8 workers) instead of a
LangGraph `Send` per table. Reasons: (1) one table's failure (an API error, a reply still invalid after the re-ask,
an unexpected exception) stays in that table's state and the run goes on, while an exception in one `Send` branch
fails the whole super-step and drops the other branches' results; (2) the worker count is the API concurrency, set
in one place; (3) every reply is cached as soon as it arrives (data/hq/cache/<prompt_hash8>/<model>/<key>-<user8>.json,
user8 = sha256 of the table's user message, so a changed table text or field summary is a miss), so a stopped run
resumes and a rerun makes no call; (4) each table still runs as its own LangGraph graph with one node per step.
The cross-encoder in `score` is shared by the workers and called under a lock (one GPU model).

filter_all (Doc2Query−−): the questions' stage 1 scores against their own table, the global bottom FILTER_FRAC
dropped, but every table keeps its MIN_KEEP best (when it has that many). Self-retrieval rank is never used to filter
(that would drop exactly the vocabulary-gap questions the index is for).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kosis_rag.llm.base import STATUS_OK
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared import tabletext as text
from kosis_rag.shared.fields import table_fields
from kosis_rag_build.hq import prompt as hq_prompt
from kosis_rag_build.hq.constants import CONCURRENCY, FILTER_FRAC, MAX_ATTEMPTS, MIN_KEEP, PROMPT_VERSION
from kosis_rag_build.hq.report import report
from kosis_rag_build.hq.rules import check_positions
from kosis_rag_build.schemas import HqBuildState, HqState, TableQuestions

REPO = Path(__file__).resolve().parents[3]
HQ_DIR = REPO / "data" / "hq"
CACHE_DIR = HQ_DIR / "cache"

PairScore = Callable[[str, list[str]], list[float]]  # (table key, questions) -> one relevance per question


def parse_reply(raw: str) -> dict:
    """The first JSON object in a reply.

    In: raw = the reply text. Out: the decoded object. Side effects: raises ValueError when there is no JSON object.
    Node: validate.
    """
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {raw[:80]!r}")
    try:
        data, _ = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"broken JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data


def model_slug(model: str) -> str:
    """The model name as a path / build id part: lower case, runs of other characters than [a-z0-9] -> '-'.

    In: model = e.g. "gpt-4o-mini". Out: e.g. "gpt-4o-mini" (at most 40 characters, "model" when nothing is left).
    Side effects: none. Node: generate (cache folder), write (build id).
    """
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:40].strip("-") or "model"


def user_sha(messages: list[dict[str, str]]) -> str:
    """sha256 of the table's user message (n2 text + field summary): what the reply was generated from.

    In: messages = prompt.build_messages(...) (the first user turn is used, so the re-ask turns do not change it).
    Out: hex digest. Side effects: none. Node: generate.
    """
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return hashlib.sha256(user.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, prompt_hash: str, model: str, key: str, user_hash: str) -> Path:
    """data/hq/cache/<prompt_hash8>/<model>/<key>-<user_hash8>.json: one file per table per prompt, model and table
    content, so a changed table (its n2 text or field summary) or another model never reuses an old reply.

    Side effects: none. Node: generate.
    """
    return cache_dir / prompt_hash[:8] / model_slug(model) / f"{key}-{user_hash[:8]}.json"


def _read_cache(path: Path, model: str, user_hash: str) -> list[dict]:
    """The cached replies when the file was written for this model and this exact user message, else [] (a miss)."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model") != model or data.get("user_sha256") != user_hash:  # another model or table text: a miss
        return []
    return data.get("responses", [])


def _write_cache(path: Path, key: str, model: str, prompt_hash: str, user_hash: str, responses: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    body = {"key": key, "model": model, "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash,
            "user_sha256": user_hash,
            "responses": [{k: v for k, v in r.items() if k != "cached"} for r in responses]}
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def table_nodes(records: dict[str, TableRecord], client: Any, *, model: str, cache_dir: Path = CACHE_DIR,
                scorer: PairScore | None = None, fmt: str = text.DOC_FORMAT) -> dict[str, Callable[[HqState], dict]]:
    """The per-table node functions, by node name: load prompt generate validate rule_check [score].

    In: records = key -> TableRecord; client = an object with generate(system, user, history) returning
        .status/.text/.usage (llm.openai.OpenAIClient; tests pass a fake); model = the model name the cache is kept under;
        cache_dir = cache root; scorer = (key, questions) -> scores (None: no score node); fmt = table text format.
    Out: {node name: step(HqState) -> update}, in graph order (graph.hq_build.build_table_graph wires them).
    Side effects: none when built; running them calls the LLM (cache misses only), writes the cache, runs the scorer.
    """
    lock = threading.Lock()

    def timed(name: str, state: HqState, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
        """The trace with this node's entry added. Node: every node."""
        return {**state.trace, name: {"seconds": time.perf_counter() - started, **extra}}

    def load(state: HqState) -> dict:
        """The table's record, fields and n2 text.

        In: state.key. Out: {"record", "fields", "doc", "trace"}. Side effects: KeyError for an unknown key.
        Node: load.
        """
        started = time.perf_counter()
        record = records[state.key]
        return {"record": record, "fields": table_fields(record), "doc": text.build_doc(record, fmt)["embed_text"],
                "trace": timed("load", state, started)}

    def prompt(state: HqState) -> dict:
        """The fixed prompt for this table.

        In: state.record, state.fields, state.doc. Out: {"messages": [system, user], "prompt_hash", "trace"}.
        Side effects: none. Node: prompt.
        """
        started = time.perf_counter()
        return {"messages": hq_prompt.build_messages(state.record, state.fields, state.doc),
                "prompt_hash": hq_prompt.prompt_hash(), "trace": timed("prompt", state, started)}

    def generate(state: HqState) -> dict:
        """One reply: from the cache when this attempt is cached, else one LLM call (then cached).

        In: state.messages (system, user, and the re-ask turns on the second attempt), state.attempts. The cache file
            is chosen by prompt, model, key and the sha256 of the first user message (cache_path).
        Out: {"raw", "responses" (+1, with cached True/False), "attempts" (+1), "trace"} or {"error"} on an API
            failure (not cached, so a rerun tries again).
        Side effects: an API call on a cache miss; writes the cache file. Node: generate.
        """
        started = time.perf_counter()
        user_hash = user_sha(state.messages)
        path = cache_path(cache_dir, state.prompt_hash, model, state.key, user_hash)
        cached = _read_cache(path, model, user_hash)
        n = state.attempts
        if len(cached) > n:
            reply = {**cached[n], "cached": True}
        else:
            msgs = state.messages
            try:
                t0 = time.perf_counter()
                result = client.generate(msgs[0]["content"], msgs[-1]["content"], history=msgs[1:-1] or None)
            except Exception as exc:  # the SDK raises after its own retries; the table is reported as failed
                return {"error": f"api: {type(exc).__name__}: {exc}", "trace": timed("generate", state, started)}
            if getattr(result, "status", STATUS_OK) != STATUS_OK:
                return {"error": f"api: {result.status}: {getattr(result, 'error', '')}",
                        "trace": timed("generate", state, started)}
            reply = {"raw": result.text or "", "usage": dict(getattr(result, "usage", None) or {}),
                     "seconds": time.perf_counter() - t0, "cached": False}
            _write_cache(path, state.key, model, state.prompt_hash, user_hash, [*cached[:n], reply])
        return {"raw": reply["raw"], "responses": [*state.responses, reply], "attempts": n + 1,
                "trace": timed("generate", state, started, cached=reply["cached"], attempt=n + 1)}

    def validate(state: HqState) -> dict:
        """Check the reply against TableQuestions; the first invalid reply gets one re-ask.

        In: state.raw, state.attempts, state.messages.
        Out: {"questions", "description"} when valid; else on the first attempt {"messages": + the reply and the
            re-ask}, on the last {"error": "invalid: ..."}; with "trace".
        Side effects: none (the re-ask call is made by generate). Node: validate.
        """
        started = time.perf_counter()
        try:
            parsed = TableQuestions.model_validate({**parse_reply(state.raw or ""), "key": state.key})
        except (ValueError, ValidationError) as exc:
            problem = str(exc).splitlines()[0][:300]
            if state.attempts < MAX_ATTEMPTS:
                reask = hq_prompt.prompt_texts().reask.format(error=problem)
                return {"messages": [*state.messages, {"role": "assistant", "content": state.raw or ""},
                                     {"role": "user", "content": reask}],
                        "trace": timed("validate", state, started, valid=False)}
            return {"error": f"invalid: {problem}", "trace": timed("validate", state, started, valid=False)}
        return {"questions": parsed.questions, "description": parsed.description.strip(),
                "trace": timed("validate", state, started, valid=True, n=len(parsed.questions))}

    def rule_check(state: HqState) -> dict:
        """Drop the questions that break a global rule (hq.rules).

        In: state.questions, state.record, state.fields. Out: {"questions": kept (reply order), "positions": their
            places in the reply, "dropped": text -> reason, "trace"}. Side effects: none. Node: rule_check.
        """
        started = time.perf_counter()
        positions, dropped = check_positions(state.questions, state.record, state.fields)
        kept = [state.questions[i] for i in positions]
        return {"questions": kept, "positions": positions, "dropped": dropped,
                "trace": timed("rule_check", state, started, kept=len(kept), dropped=len(dropped))}

    def score(state: HqState) -> dict:
        """Score each kept question against its own table with the stage 1 cross-encoder.

        In: state.key, state.questions. Out: {"scores": one per question, "trace"}.
        Side effects: cross-encoder inference (under the shared lock). Node: score.
        """
        started = time.perf_counter()
        if not state.questions:
            return {"scores": [], "trace": timed("score", state, started)}
        with lock:
            scores = [float(v) for v in scorer(state.key, [q.text for q in state.questions])]
        return {"scores": scores, "trace": timed("score", state, started)}

    nodes = {"load": load, "prompt": prompt, "generate": generate, "validate": validate, "rule_check": rule_check}
    if scorer is not None:
        nodes["score"] = score
    return nodes


def build_id(prompt_hash: str, tables_sha: str, model: str, frac: float = FILTER_FRAC, min_keep: int = MIN_KEEP,
             limit: int | None = None) -> str:
    """hq-<prompt8>-<tables8>-<model>-f<percent dropped>[-k<min_keep>][-n<limit>]: the same inputs always give the
    same id, and builds that differ in model, filter share or min_keep get their own folder and Chroma collection.

    In: prompt_hash = prompt.prompt_hash(); tables_sha = sha256 of tables.jsonl; model = generation model (as
        model_slug); frac = score filter share (0 = off: then min_keep does nothing and is left out, "-f0");
        min_keep = questions every table keeps; limit = first N tables only (a sample build), None = all.
    Out: the build id (also in the Chroma name), e.g. hq-1a2b3c4d-5e6f7a8b-gpt-4o-mini-f30-k2.
    Side effects: none. Node: write.
    """
    pct = round(frac * 100)
    keep = f"-k{min_keep}" if pct > 0 else ""
    return f"hq-{prompt_hash[:8]}-{tables_sha[:8]}-{model_slug(model)}-f{pct}{keep}" + (f"-n{limit}" if limit else "")


def filter_scores(items: list[dict], frac: float = FILTER_FRAC, min_keep: int = MIN_KEEP
                  ) -> tuple[list[dict], list[dict]]:
    """Doc2Query−− filter over every table's questions at once.

    In: items = {"id", "key", "pos", "score" (float or None), ...} per question; frac = share of the scored questions
        in the global bottom; min_keep = best questions per table that are never dropped.
    Out: (kept items in input order, dropped items). The bottom = the round(frac * n) lowest scores (ties: key, then
        position); a table's min_keep best are kept even when they sit in the bottom, so fewer than frac may go.
        Items without a score, and everything when frac <= 0, are kept.
    Side effects: none. Node: filter_all.
    """
    scored = [it for it in items if it.get("score") is not None]
    if frac <= 0 or not scored:
        return list(items), []
    by_key: dict[str, list[dict]] = {}
    for it in scored:
        by_key.setdefault(it["key"], []).append(it)
    protected = {it["id"] for group in by_key.values()
                 for it in sorted(group, key=lambda x: (-x["score"], x["pos"]))[:min_keep]}
    bottom = sorted(scored, key=lambda x: (x["score"], x["key"], x["pos"]))[: round(frac * len(scored))]
    drop = {it["id"] for it in bottom if it["id"] not in protected}
    return [it for it in items if it["id"] not in drop], [it for it in bottom if it["id"] in drop]


def build_nodes(table_graph: Any, *, out_root: Path = HQ_DIR, concurrency: int = CONCURRENCY,
                log: Callable[[str], None] = print) -> dict[str, Callable[[HqBuildState], dict]]:
    """The whole-build node functions, by node name: generate_all filter_all write.

    In: table_graph = a compiled graph.hq_build.build_table_graph(...) (anything with invoke({"key"})); out_root =
        data/hq; concurrency = worker threads (API concurrency); log = progress output.
    Out: {node name: step(HqBuildState) -> update} (graph.hq_build.build_build_graph wires them).
    Side effects: none when built; running them runs every table graph and writes data/hq/<build_id>/.
    """

    def timed(name: str, state: HqBuildState, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
        return {**state.trace, name: {"seconds": time.perf_counter() - started, **extra}}

    def one(key: str) -> HqState:
        try:
            return HqState.model_validate(table_graph.invoke({"key": key}))
        except Exception as exc:  # anything unexpected stays this table's failure
            return HqState(key=key, error=f"crash: {type(exc).__name__}: {exc}")

    def generate_all(state: HqBuildState) -> dict:
        """Run the table graph for every key, `concurrency` at a time.

        In: state.keys. Out: {"results": one HqState per key, in key order, "trace"}.
        Side effects: LLM calls on cache misses, cache writes, scorer inference; progress lines via log.
        Node: generate_all.
        """
        started = time.perf_counter()
        results: list[HqState] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for n, res in enumerate(pool.map(one, state.keys), 1):  # map keeps the key order
                results.append(res)
                if n % 100 == 0 or n == len(state.keys):
                    failed = sum(r.error is not None for r in results)
                    log(f"  {n}/{len(state.keys)} tables ({failed} failed, {time.perf_counter() - started:.0f}s)")
        return {"results": results, "trace": timed("generate_all", state, started)}

    def filter_all(state: HqBuildState) -> dict:
        """The global score filter over every table's rule-checked questions (filter_scores).

        In: state.results, state.frac, state.min_keep. Out: {"items", "kept", "score_dropped", "trace"}.
        Side effects: none. Node: filter_all.
        """
        started = time.perf_counter()
        items = []
        for r in state.results:
            for n, q in enumerate(r.questions):
                pos = r.positions[n] if n < len(r.positions) else n
                items.append({"id": f"{r.key}#q{pos}", "key": r.key, "pos": pos, "type": q.type, "text": q.text.strip(),
                              "score": r.scores[n] if n < len(r.scores) else None})
        kept, dropped = filter_scores(items, state.frac, state.min_keep)
        return {"items": items, "kept": kept, "score_dropped": dropped,
                "trace": timed("filter_all", state, started, kept=len(kept), dropped=len(dropped))}

    def write(state: HqBuildState) -> dict:
        """Write data/hq/<build_id>/: questions.jsonl, descriptions.jsonl, failed.jsonl, build.json, report.md.

        In: the whole state. Out: {"build_id", "out_dir", "totals", "trace"}.
        Side effects: creates the folder and writes the files (a rerun of the same build overwrites them). Node: write.
        """
        started = time.perf_counter()
        bid = build_id(hq_prompt.prompt_hash(), state.tables_sha, state.model, state.frac, state.min_keep, state.limit)
        out = out_root / bid
        out.mkdir(parents=True, exist_ok=True)
        totals = build_totals(state)
        with (out / "questions.jsonl").open("w", encoding="utf-8") as f:
            for it in state.kept:
                f.write(json.dumps({k: it[k] for k in ("id", "key", "type", "text")}, ensure_ascii=False) + "\n")
        with (out / "descriptions.jsonl").open("w", encoding="utf-8") as f:
            for r in state.results:
                if r.description:
                    f.write(json.dumps({"key": r.key, "description": r.description}, ensure_ascii=False) + "\n")
        with (out / "failed.jsonl").open("w", encoding="utf-8") as f:
            for r in state.results:
                if r.error:
                    f.write(json.dumps({"key": r.key, "error": r.error}, ensure_ascii=False) + "\n")
        manifest = {"build_id": bid, "prompt_version": PROMPT_VERSION, "prompt_hash": hq_prompt.prompt_hash(),
                    "model": state.model, "tables_sha256": state.tables_sha, "filter_frac": state.frac,
                    "min_keep": state.min_keep, "limit": state.limit, "totals": totals}
        (out / "build.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "report.md").write_text(report(state, bid, totals), encoding="utf-8")
        return {"build_id": bid, "out_dir": str(out), "totals": totals, "trace": timed("write", state, started)}

    return {"generate_all": generate_all, "filter_all": filter_all, "write": write}


def build_totals(state: HqBuildState) -> dict[str, Any]:
    """Counts and token totals of a build (all replies, and the ones this run called for).

    In: state after filter_all. Out: dict (tables, failed, generated, rule_dropped, score_dropped, kept, calls,
        cached, input/output tokens all and new, api_seconds). Side effects: none. Node: write.
    """
    replies = [rep for r in state.results for rep in r.responses]
    new = [rep for rep in replies if not rep.get("cached")]

    def tok(reps: list[dict], name: str) -> int:
        return sum(int((rep.get("usage") or {}).get(name) or 0) for rep in reps)

    return {
        "tables": len(state.results), "failed": sum(r.error is not None for r in state.results),
        "generated": sum(len(r.questions) + len(r.dropped) for r in state.results),
        "rule_dropped": sum(len(r.dropped) for r in state.results), "score_dropped": len(state.score_dropped),
        "kept": len(state.kept), "tables_with_questions": len({it["key"] for it in state.kept}),
        "replies": len(replies), "calls": len(new), "cached": len(replies) - len(new),
        "input_tokens": tok(replies, "input_tokens"), "output_tokens": tok(replies, "output_tokens"),
        "new_input_tokens": tok(new, "input_tokens"), "new_output_tokens": tok(new, "output_tokens"),
        "api_seconds": round(sum(float(rep.get("seconds") or 0) for rep in new), 1),
    }


def tables_sha(path: Path) -> str:
    """sha256 of tables.jsonl (part of the build id).

    Side effects: reads the file. Node: none (run before the build graph).
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_scorer(ckpt: str, records: list, fmt: str, device: str | None = None,
                log: Callable[[str], None] = print) -> PairScore:
    """(key, questions) -> stage 1 scores of each question against the table's own n2 passage.

    In: ckpt = fine-tuned cross-encoder (.pt or run folder); records = every table; fmt = its text format;
        device = cpu / cuda (default: auto). Out: the PairScore the score node calls.
    Side effects: loads the cross-encoder (GPU memory). Node: score (built here, called there).
    """
    from kosis_rag.adapters.models import load_cross_encoder
    from kosis_rag.rerank.stage1 import passage_scorer

    text_of = {r.info.key: text.build_doc(r, fmt)["embed_text"] for r in records}
    scorer = load_cross_encoder(ckpt, device=device, doc_format=fmt)
    score = passage_scorer(scorer, text_of, text.norm_query)  # the same scorer and question rules as rerank1
    log(f"stage 1 scorer: {scorer.ckpt}")
    return lambda key, questions: [score(q, [key])[0] for q in questions]
