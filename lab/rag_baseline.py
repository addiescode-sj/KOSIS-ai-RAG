"""Master's RAG baseline end to end, without Docker: ChromaDB + BM25 hybrid -> answer -> HTML views.

Master's own pieces, unchanged: index.to_document / kure_embedder / load (T2-24, KURE-v1, labels capped at 100),
BM25 over TableRecord.search_text (bm25-record), chain.answer (T2-18/T2-26) with the Phase 1 LLM client
(vendor/phase1/configs/model.yaml, HCX-DASH-002), and the L1 harness (evals.run_eval -> report.md / report.html).
Added here only:
    - an in-memory chromadb client loaded from cached vectors in place of the docker server; same collection API
    - an RRF hybrid of the two (c=60, top 50 each), since T2-06 is not on master yet
    - answers.html: question -> top 5 tables (gold marked) -> the chain's answer, for L1 questions

    python lab/rag_baseline.py                        # build the collection once, then L1 test: bm25 / chroma / hybrid
    python lab/rag_baseline.py --answers 20           # + hybrid answers for the first 20 L1 test questions (HCX)
    python lab/rag_baseline.py --golden phase1 --answers all   # Phase 1 golden (90 questions, 10 tables)
    python lab/rag_baseline.py --skip-l1 --answers all --backend mock
    python lab/rag_baseline.py --e2e all              # Phase 1 golden end to end: hybrid top 1 -> summarize_stat -> L3 grade
    python lab/rag_baseline.py --dense bge-m3                 # lab V0 dense (bge-m3, n2 text) in place of KURE-v1
    python lab/rag_baseline.py --dense ft --ckpt data/lab/ft/<run>/seen-e2.pt   # plan 29 fine-tuned weights

--dense bge-m3 / ft embed the lab V0 text (n2 rules, labels > 60 summarized; the text the weights were trained on)
and n2-normalize the question; BM25 stays master's bm25-record either way, so only the dense side changes.

The L1 golden set is evals/datasets/retrieval_golden.jsonl (gitignored); when it is missing it is copied from
eval/golden/retrieval_golden.jsonl. The ops history line goes to data/lab/rag_baseline/ (not the tracked
evals/reports/ops_history.jsonl). Outputs: evals/reports/<run>/report.html and data/lab/rag_baseline/<stamp>/answers.html.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import html
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("vendor/phase1/src", "vendor/phase1", "src", "."):  # same paths as pytest.ini, "." first
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))
os.environ.setdefault("KOSIS_WORKSPACE", str(ROOT))  # Phase 1 reads .env under this root (docs/ISSUES.md P2-07)

from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

OUT = ROOT / "data" / "lab" / "rag_baseline"
CHROMA_DIR = ROOT / "chroma"
OLD_GOLDEN = ROOT / "eval" / "golden" / "retrieval_golden.jsonl"
PHASE1_GOLDEN = ROOT / "eval" / "golden" / "golden.jsonl"
KIND: dict[str, str] = {}  # qid -> group shown in the views (Phase 1 kind with --golden phase1)
POOL = 50  # candidates per retriever before fusion
RRF_C = 60


class RRFRetriever(BaseRetriever):
    """Reciprocal rank fusion by table key: score = sum over retrievers of 1 / (c + rank)."""

    retrievers: list[BaseRetriever]
    c: int = RRF_C
    k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        score: dict[str, float] = {}
        first: dict[str, Document] = {}
        for r in self.retrievers:
            for rank, d in enumerate(r.invoke(query), 1):
                key = d.metadata["key"]
                score[key] = score.get(key, 0.0) + 1.0 / (self.c + rank)
                first.setdefault(key, d)
        top = sorted(score, key=lambda k: (-score[k], k))[: self.k]  # ties by key, so runs repeat
        return [Document(page_content=first[k].page_content, metadata={**first[k].metadata, "score": score[k]},
                         id=first[k].id) for k in top]


def embed_all(embed, docs: list[str]) -> list[list[float]]:
    """Like index.main (similar lengths per batch), but a batch that runs out of GPU memory is redone one by one."""
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    vectors: list = [None] * len(docs)
    started = time.perf_counter()
    for b in range(0, len(order), 64):
        chunk = order[b:b + 64]
        try:
            out = embed([docs[i] for i in chunk])
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            import torch

            torch.cuda.empty_cache()
            out = [embed([docs[i]])[0] for i in chunk]
        for i, v in zip(chunk, out):
            vectors[i] = v
        print(f"  embedded {min(b + 64, len(docs))}/{len(docs)} ({time.perf_counter() - started:.0f}s)", flush=True)
    return vectors


def old_persistent_vectors(name: str, ids: list[str]) -> list | None:
    """Vectors from the collection an earlier version of this script left in chroma/, if it can still be read."""
    try:
        import chromadb
        from chromadb.config import Settings

        got = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)) \
            .get_collection(name).get(ids=ids, include=["embeddings"])
        by_id = dict(zip(got["ids"], got["embeddings"]))
        return [list(map(float, by_id[i])) for i in ids] if len(by_id) == len(ids) else None
    except Exception as exc:  # broken HNSW files etc.: embed again
        print(f"  (old chroma/ collection not usable: {type(exc).__name__})")
        return None


def dense_kure(records, device: str | None) -> dict:
    """Master's dense side: KURE-v1 over index.to_document (T2-24)."""
    from kosis_rag import index

    embed, _, _ = index.kure_embedder(device)
    return {"label": "KURE-v1 (master)", "tag": index.EMBED_MODEL, "docs": [index.to_document(r) for r in records],
            "embed_docs": embed, "embed_query": lambda q: embed([q])[0], "legacy": True}


def dense_bge(records, ckpt: Path | None) -> dict:
    """bge-m3 as in the lab V0 (n2 text, labels > 60 summarized, n2 query), with plan 29 weights when ckpt is given."""
    import numpy as np
    from lab_common import DEFAULT_MODEL, docs, encode, load_model, query_text
    from lab_common import load_records as lab_records

    text = {d["key"]: d["embed"] for d in docs(lab_records(), "norm", 60)}
    if ckpt:
        from ft_train import load_ft

        model = load_ft(DEFAULT_MODEL, ckpt, 1024)
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:8]
        tag, label = f"{DEFAULT_MODEL}+ft:{ckpt.parent.name}/{ckpt.stem}@{sha}", f"bge-m3 fine-tuned {ckpt.parent.name}/{ckpt.stem}"
    else:
        model = load_model(DEFAULT_MODEL, 1024)
        tag, label = DEFAULT_MODEL, "bge-m3 (lab V0)"

    def run(texts: list[str], role: str) -> list[list[float]]:
        return np.asarray(encode(model, DEFAULT_MODEL, texts, role, 16), dtype=np.float32).tolist()

    return {"label": label, "tag": tag, "docs": [text[r.info.key] for r in records],
            "embed_docs": lambda t: run(t, "passage"), "embed_query": lambda q: run([query_text(q, "norm")], "query")[0]}


def chroma_retriever(records, dense: dict, k: int):
    """Chroma collection for this tables.jsonl and embedding model, loaded into an in-memory client each run.

    The vectors are embedded once and cached as .npy under data/lab/rag_baseline/. The collection itself is not
    kept on disk: a PersistentClient folder under OneDrive left HNSW segments that could not be read back
    ("Error loading hnsw index"), and loading 2,364 cached vectors takes a second.
    """
    import chromadb
    import numpy as np
    from chromadb.config import Settings

    from kosis_rag import index
    from kosis_rag.retriever import ChromaTableRetriever

    index.EMBED_MODEL = dense["tag"]  # recorded in the collection metadata and checked below
    data_sha = hashlib.sha256(index.REPO.joinpath("data", "tables.jsonl").read_bytes()).hexdigest()
    docs = dense["docs"]
    # the cache is keyed by the model tag and the exact passages, so another model or document text never reuses it
    model_tag = hashlib.sha256("\n".join([index.EMBED_MODEL, *docs]).encode("utf-8")).hexdigest()[:8]
    name = f"local_{data_sha[:8]}_{model_tag}"
    ids = [r.info.key for r in records]
    metas = [index.to_metadata(r, name) for r in records]
    npy, keys_file = OUT / f"{name}.npy", OUT / f"{name}.keys.json"
    # earlier versions of this script named the KURE-v1 cache/collection by the model alone
    legacy = f"local_{index.COLLECTION_PREFIX}{data_sha[:8]}_{hashlib.sha256(index.EMBED_MODEL.encode()).hexdigest()[:6]}"
    vectors = None
    for stem in [name, *([legacy] if dense.get("legacy") else [])]:
        f, kf = OUT / f"{stem}.npy", OUT / f"{stem}.keys.json"
        if f.exists() and kf.exists() and json.loads(kf.read_text(encoding="utf-8")) == ids:
            vectors = np.load(f).tolist()
            print(f"collection {name}: cached vectors from {f.name} ({len(ids)} tables)")
            break
    if vectors is None and dense.get("legacy"):
        vectors = old_persistent_vectors(legacy, ids)
    if vectors is None:
        print(f"collection {name}: embedding {len(docs)} tables with {index.EMBED_MODEL} (once)")
        vectors = embed_all(dense["embed_docs"], docs)
    np.save(npy, np.asarray(vectors, dtype=np.float32))
    keys_file.write_text(json.dumps(ids), encoding="utf-8")

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    record = index.build_record(name, data_sha, len(docs), len(vectors[0]), 0, 0)
    # HNSW is approximate and a rebuilt index can return a different top 10 (seen in the smoke test);
    # with ef >= number of tables the search is effectively exact, so runs repeat. 2,364 vectors: still fast.
    record |= {"hnsw:construction_ef": 200, "hnsw:search_ef": max(100, len(ids))}
    count = index.load(client, ids, docs, metas, vectors, name, record)
    print(f"collection {name}: {count} tables in Chroma (in memory)")
    collection = client.get_collection(name, embedding_function=None)
    assert (collection.metadata or {}).get("embed_model") == index.EMBED_MODEL
    return ChromaTableRetriever(collection=collection, embed_query=dense["embed_query"], k=k)


def build_retrievers(device: str | None, kind: str, ckpt: Path | None) -> tuple[dict[str, BaseRetriever], str]:
    """bm25-record, chroma-<kind>, hybrid-<kind> (the names become the report stages), and the dense label."""
    from evals.runners.retrieval import bm25_retrievers, load_records

    records = load_records()
    bm25 = bm25_retrievers(records, POOL)["bm25-record"]
    dense = dense_kure(records, device) if kind == "kure" else dense_bge(records, ckpt if kind == "ft" else None)
    chroma = chroma_retriever(records, dense, POOL)
    return {"bm25-record": bm25, f"chroma-{kind}": chroma, f"hybrid-{kind}": RRFRetriever(retrievers=[bm25, chroma])}, \
        dense["label"]


def ensure_golden() -> Path:
    from evals.schemas import GOLDEN

    if not GOLDEN.exists():
        if not OLD_GOLDEN.exists():
            raise FileNotFoundError(f"{GOLDEN} is missing (and no {OLD_GOLDEN} to copy)")
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD_GOLDEN, GOLDEN)
        print(f"copied {OLD_GOLDEN.relative_to(ROOT)} -> {GOLDEN.relative_to(ROOT)}")
    return GOLDEN


def use_phase1_golden() -> Path:
    """Phase 1 golden (90 questions, each written on one of 10 tables) as L1 items: that table is the one slot.

    Every item is kept, refusals and clarifications too: the question is still about its table, so finding it
    is right. All go to split test; the group in the views is the Phase 1 kind (answerable, not_answerable, ...).
    """
    from evals import report_html, run_eval, schemas

    items = [json.loads(line) for line in PHASE1_GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    path = OUT / "phase1_l1.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            KIND[it["id"]] = f"p1-{it['kind']}"
            row = {"qid": it["id"], "split": "test", "qtype": "exact_name", "source": "phase1", "question": it["question"],
                   "required": [[Path(it["table"]).stem]], "phase1_id": it["id"], "notes": f"phase1 kind {it['kind']}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    schemas.GOLDEN = run_eval.GOLDEN = path
    report_html.QTYPE_ORDER = ["all", *sorted(set(KIND.values()))]
    print(f"Phase 1 golden: {len(items)} questions on {len({Path(i['table']).stem for i in items})} tables -> {path.relative_to(ROOT)}")
    return path


def run_l1(retrievers: dict[str, BaseRetriever], split: str, tau: float | None = None) -> list[Path]:
    """evals.run_eval as is; only build() is pointed at the retrievers above and the history file is moved."""
    from evals import ops, report_html, run_eval
    from evals.runners import retrieval as runner

    history = OUT / "ops_history.jsonl"
    ops.append_history = functools.partial(ops.append_history, path=history)
    report_html.read_history = functools.partial(ops.read_history, path=history)
    runner.build = lambda name, items, k: retrievers[name]
    if KIND:  # group rows by the Phase 1 kind instead of the placeholder qtype
        run = runner.run
        runner.run = lambda r, items, k: [{**row, "qtype": KIND.get(row["qid"], row["qtype"])} for row in run(r, items, k)]
    before = set(run_eval.REPORTS.glob("*"))
    for name in retrievers:
        extra = ["--tau", str(tau)] if tau is not None else []
        run_eval.main(["--layer", "l1", "--retriever", name, "--stage", f"base-{name}", "--split", split, *extra])
        time.sleep(1.1)  # run ids are per second
    return sorted(set(run_eval.REPORTS.glob("*")) - before)


def make_client(backend: str | None):
    """Phase 1 LLM client. Phase 1 reads secrets from the work root's .env (this repo); a key that is only in
    vendor/phase1/.env is picked up too (values already set are kept, nothing is printed)."""
    from dotenv import load_dotenv
    from kosis_chat.config import load_settings
    from kosis_chat.llm.model import build_client

    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / "vendor" / "phase1" / ".env", override=False)
    settings = load_settings(backend=backend)
    return settings, build_client(settings)  # fails here, before any retrieval work, when the key is missing


def answers(retriever: BaseRetriever, llm, n: str, split: str, interval: float, dense_label: str) -> Path:
    from kosis_rag import chain
    from evals import schemas

    settings, client = llm
    items = [i for i in schemas.load_items(schemas.GOLDEN) if split == "all" or i.split == split]
    items = items if n == "all" else items[: int(n)]
    rows = []
    for j, item in enumerate(items, 1):
        started = time.perf_counter()
        try:
            res = chain.answer(retriever, client, item.question, k=5)
            res_d, error = res.model_dump(), None
        except Exception as exc:  # one failing question is shown, the run goes on
            res_d, error = {"candidates": [], "answer": "", "answer_origin": "error", "no_answer": False,
                            "llm_status": None}, f"{type(exc).__name__}: {exc}"
        slots = [list(s) for s in item.required]
        keys = [c["key"] for c in res_d["candidates"]]
        rank = [next((r for r, k in enumerate(keys, 1) if k in s), None) for s in slots]
        rows.append({"qid": item.qid, "qtype": KIND.get(item.qid, item.qtype), "question": item.question, "answerable": item.answerable,
                     "slots": slots, "slot_rank": rank, "hit5": bool(slots) and all(r is not None for r in rank),
                     "seconds": round(time.perf_counter() - started, 2), "error": error, **res_d})
        print(f"  [{j}/{len(items)}] {item.qid} {res_d['answer_origin']} hit@5={rows[-1]['hit5']}", flush=True)
        if interval and settings.backend != "mock" and j < len(items):
            time.sleep(interval)
    out = OUT / f"{dt.datetime.now():%Y%m%d-%H%M%S}-answers"
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                                       encoding="utf-8")
    page = out / "answers.html"
    page.write_text(answers_html(rows, f"{settings.backend} · {settings.model_name}", split, dense_label), encoding="utf-8")
    return page

def e2e(retriever: BaseRetriever, backend: str | None, n: str, interval: float, dense_label: str) -> Path:
    """Phase 1 golden end to end: the hybrid's top 1 table -> Phase 1 summarize_stat -> the team's L3 grader.

    Same grader and rules as `run_eval --layer l3` (evals/evaluators/calculation.py), graded against the golden
    table's facts and Parquet cells, so an answer read from another table fails unless its numbers are the same.
    Cause per row: retrieval (top 1 is not the golden table) or answer (right table, wrong answer).
    """
    from kosis_chat.api import summarize_stat
    from kosis_chat.config import load_settings

    from evals.builders.phase1 import convert
    from evals.evaluators import calculation
    from evals.ops import usage_client

    items = [convert(json.loads(line)) for line in PHASE1_GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = items if n == "all" else items[: int(n)]
    settings = load_settings(backend=backend)
    client = usage_client(settings)
    rows = []
    for j, item in enumerate(items, 1):
        keys = [d.metadata["key"] for d in retriever.invoke(item.question)]
        used = keys[0] if keys else None
        client.reset()
        started = time.perf_counter()
        answer, error = None, None
        if used is None:
            error = "no table retrieved"
        else:
            try:
                answer = summarize_stat(item.question, used, client=client, retry_interval=interval)
            except Exception as exc:  # recorded as run_failure by the grader, the run goes on
                error = f"{type(exc).__name__}: {exc}"
        cells = calculation.read_cells(ROOT / "data" / "values" / f"{item.table_key}.parquet")
        g = calculation.grade(item, answer, cells, error)
        right_table = used == item.table_key
        cause = "pass" if g.verdict == "PASS" else ("answer" if right_table else "retrieval")
        rows.append({**g.model_dump(), "question": item.question, "gold_table": item.table_key, "used_table": used,
                     "right_table": right_table, "gold_rank": keys.index(item.table_key) + 1 if item.table_key in keys else None,
                     "top5": keys[:5], "cause": cause, "answer_text": answer.text if answer else "", "error": error,
                     "seconds": round(time.perf_counter() - started, 1), "llm_calls": client.calls,
                     "input_tokens": client.input_tokens, "output_tokens": client.output_tokens})
        print(f"  [{j}/{len(items)}] {item.qid} table {'ok' if right_table else 'WRONG'} -> {g.verdict}", flush=True)
        if interval and settings.backend != "mock" and j < len(items):
            time.sleep(interval)
    out = OUT / f"{dt.datetime.now():%Y%m%d-%H%M%S}-e2e"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                                       encoding="utf-8")
    report = e2e_report(rows, f"{settings.backend} · {settings.model_name}", dense_label)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    page = out / "e2e.html"
    page.write_text(e2e_html(rows, f"{settings.backend} · {settings.model_name}", dense_label, report), encoding="utf-8")
    return page


E2E_KINDS = ["all", "answerable", "not_answerable", "clarify", "default_allowed"]


def e2e_report(rows: list[dict], model: str, dense_label: str) -> str:
    def count(sel: list[dict]) -> str:
        v = {x: sum(r["verdict"] == x for r in sel) for x in ("PASS", "PARTIAL", "FAIL")}
        return f"{len(sel)} | {v['PASS']} | {v['PARTIAL']} | {v['FAIL']}"

    lines = [f"# End to end — Phase 1 golden ({len(rows)} questions)", "",
             f"- hybrid (BM25 bm25-record + Chroma {dense_label}) top 1 -> summarize_stat ({model}) -> L3 grader", "",
             "| group | n | PASS | PARTIAL | FAIL |", "| --- | --- | --- | --- | --- |"]
    for k in E2E_KINDS:
        sel = rows if k == "all" else [r for r in rows if r["kind"] == k]
        if sel:
            lines.append(f"| {'all' if k == 'all' else 'kind=' + k} | {count(sel)} |")
    lines.append(f"| top 1 = golden table | {count([r for r in rows if r['right_table']])} |")
    lines.append(f"| top 1 = other table | {count([r for r in rows if not r['right_table']])} |")
    causes = {c: sum(r["cause"] == c for r in rows) for c in ("pass", "retrieval", "answer")}
    errors: dict[str, int] = {}
    for r in rows:
        for e in r["errors"]:
            errors[e] = errors.get(e, 0) + 1
    lines += ["", f"- table found at top 1: {sum(r['right_table'] for r in rows)}/{len(rows)}",
              f"- not PASS because of retrieval (other table at top 1): {causes['retrieval']} · "
              f"because of the answer (right table): {causes['answer']}",
              "- error types: " + (", ".join(f"{k} {v}" for k, v in sorted(errors.items(), key=lambda x: -x[1])) or "none"),
              f"- LLM calls {sum(r['llm_calls'] for r in rows)} · tokens in {sum(r['input_tokens'] for r in rows):,} "
              f"out {sum(r['output_tokens'] for r in rows):,}"]
    return "\n".join(lines) + "\n"


def e2e_html(rows: list[dict], model: str, dense_label: str, report: str) -> str:
    e = html.escape
    n = len(rows)
    tiles = [("PASS", f"{sum(r['verdict'] == 'PASS' for r in rows)}/{n}"),
             ("table at top 1", f"{sum(r['right_table'] for r in rows)}/{n}"),
             ("fail: retrieval", sum(r["cause"] == "retrieval" for r in rows)),
             ("fail: answer", sum(r["cause"] == "answer" for r in rows))]
    body = []
    for r in rows:
        cls = {"PASS": "ok", "PARTIAL": "na", "FAIL": "no"}[r["verdict"]]
        table = (f'<span class="tag ok">✓ {e(r["used_table"] or "-")}</span>' if r["right_table"] else
                 f'<span class="tag no">✕ {e(r["used_table"] or "-")}</span><br><span class="m">gold rank '
                 f'{r["gold_rank"] or "-"}</span>')
        detail = e("; ".join(r["details"][:6]) or (r["error"] or ""))
        body.append(
            f'<tr data-kind="{e(r["cause"])}"><td class="m">{e(r["qid"])}<br>{e(r["kind"])}</td><td>{e(r["question"])}</td>'
            f'<td class="m">{e(r["gold_table"])}</td><td>{table}</td>'
            f'<td><span class="tag {cls}">{r["verdict"]}</span><br><span class="m">{e(", ".join(r["errors"]))}</span></td>'
            f'<td><details><summary class="m">{e((r["answer_text"] or r["error"] or "")[:60])}…</summary>'
            f'<pre>{e(r["answer_text"] or "")}</pre><p class="m">{detail}</p></details></td></tr>')
    tile_html = "".join(f'<div class="tile"><span class="m">{e(k)}</span><b>{e(str(v))}</b></div>' for k, v in tiles)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG end to end</title>
<style>{CSS}
.wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:860px;background:var(--card)}}
th,td{{border-bottom:1px solid var(--border);padding:8px;text-align:left;vertical-align:top}}th{{font-size:12px;color:var(--muted)}}
details{{border:0;padding:0;margin:0;background:none}}main{{max-width:1200px}}</style></head><body><main>
<h1>RAG end to end — Phase 1 golden</h1>
<p class="sub">hybrid (BM25 bm25-record + Chroma {e(dense_label)}) top 1 → summarize_stat → L3 grader · {e(model)} ·
{dt.datetime.now():%Y-%m-%d %H:%M}</p>
<div class="tiles">{tile_html}</div>
<details><summary>report.md</summary><pre>{e(report)}</pre></details>
<div class="filters"><button class="on" data-f="all">all</button><button data-f="retrieval">fail: retrieval</button>
<button data-f="answer">fail: answer</button><button data-f="pass">pass</button></div>
<div class="wrap"><table><thead><tr><th>id · kind</th><th>질문</th><th>정답 표</th><th>사용한 표 (1위)</th><th>판정</th><th>답변</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
</main><script>
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('.filters button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('tbody tr').forEach(d=>d.style.display=(b.dataset.f==='all'||d.dataset.kind===b.dataset.f)?'':'none');}});
</script></body></html>"""



CSS = """
:root{color-scheme:light;--bg:#f5f5f3;--card:#fcfcfb;--border:#e2e1dc;--text:#0b0b0b;--muted:#6b6a65;
--good:#0c7a0c;--good-bg:#e5f3e5;--bad:#b42f2f;--bad-bg:#f8e6e6;--warn:#8a5a00;--warn-bg:#fbf0d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;--card:#1a1a19;
--border:#33332f;--text:#f0efec;--muted:#a3a29b;--good:#5cc25c;--good-bg:#173017;--bad:#e07070;--bad-bg:#361a1a;
--warn:#e0b050;--warn-bg:#352a12}}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,-apple-system,"Malgun Gothic",sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 48px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted)}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.tile{background:var(--card);border:1px solid var(--border);
border-radius:10px;padding:10px 14px;min-width:120px}.tile b{display:block;font-size:20px}
.filters{margin:8px 0 16px;display:flex;gap:8px;flex-wrap:wrap}.filters button{border:1px solid var(--border);
background:var(--card);color:var(--text);border-radius:999px;padding:4px 12px;cursor:pointer}
.filters button.on{border-color:var(--text)}
details{background:var(--card);border:1px solid var(--border);border-radius:10px;margin:8px 0;padding:10px 14px}
summary{cursor:pointer;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}summary .q{flex:1 1 300px}
.tag{font-size:12px;border-radius:6px;padding:1px 7px;white-space:nowrap}.ok{color:var(--good);background:var(--good-bg)}
.no{color:var(--bad);background:var(--bad-bg)}.na{color:var(--warn);background:var(--warn-bg)}.m{color:var(--muted)}
ol{margin:8px 0;padding-left:22px}li.gold{font-weight:600;color:var(--good)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border-radius:8px;padding:10px;margin:8px 0 0;
font:13px/1.55 inherit}
"""


def answers_html(rows: list[dict], model: str, split: str, dense_label: str) -> str:
    e = html.escape
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    origins: dict[str, int] = {}
    for r in rows:
        origins[r["answer_origin"]] = origins.get(r["answer_origin"], 0) + 1
    hit = sum(r["hit5"] for r in ans)
    refused = sum(bool(r["no_answer"]) for r in una)
    tiles = [("questions", len(rows)), ("Hit@5 (answerable)", f"{hit}/{len(ans)}" if ans else "-"),
             ("refused (no answer)", f"{refused}/{len(una)}" if una else "-")]
    tiles += [(f"origin: {k}", v) for k, v in sorted(origins.items())]
    cards = []
    for r in rows:
        gold = {k for s in r["slots"] for k in s}
        if not r["answerable"]:
            verdict = ("ok", "refused ✓") if r["no_answer"] else ("no", "should refuse ✕")
        else:
            verdict = ("ok", "hit@5 ✓") if r["hit5"] else ("no", "miss ✕")
        kind = "unans" if not r["answerable"] else ("hit" if r["hit5"] else "miss")
        cands = "".join(
            f'<li class="{"gold" if c["key"] in gold else ""}">{e(c["table_name"])} <span class="m">'
            f'{e(c["tbl_id"])} · {e(c["organization"])} · {e(c["period"])} · score {c["score"] or 0:.4f}</span></li>'
            for c in r["candidates"])
        golds = " / ".join(" | ".join(s) for s in r["slots"]) or "(none: should refuse)"
        ranks = ", ".join("-" if x is None else str(x) for x in r["slot_rank"])
        err = f'<p class="no">error: {e(r["error"])}</p>' if r["error"] else ""
        cards.append(
            f'<details data-kind="{kind}"><summary><span class="tag {verdict[0]}">{verdict[1]}</span>'
            f'<span class="q">{e(r["question"])}</span><span class="m">{e(r["qid"])} · {e(r["qtype"])} · '
            f'{e(r["answer_origin"])}</span></summary>'
            f'<p class="m">gold: {e(golds)} · rank per slot: {ranks} · llm {e(str(r["llm_status"]))} · {r["seconds"]}s</p>'
            f'<ol>{cands or "<li class=m>(no candidates)</li>"}</ol>{err}<pre>{e(r["answer"] or "")}</pre></details>')
    tile_html = "".join(f'<div class="tile"><span class="m">{e(k)}</span><b>{e(str(v))}</b></div>' for k, v in tiles)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG baseline answers</title>
<style>{CSS}</style></head><body><main>
<h1>RAG baseline answers</h1>
<p class="sub">hybrid (BM25 bm25-record + Chroma {e(dense_label)}, RRF c={RRF_C}) → top 5 → chain.answer · {e(model)} ·
{"Phase 1 golden" if KIND else "L1 " + e(split)} · {dt.datetime.now():%Y-%m-%d %H:%M}</p>
<div class="tiles">{tile_html}</div>
<div class="filters"><button class="on" data-f="all">all</button><button data-f="miss">miss</button>
<button data-f="hit">hit</button><button data-f="unans">no-answer items</button></div>
{''.join(cards)}
</main><script>
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('.filters button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('details').forEach(d=>d.style.display=(b.dataset.f==='all'||d.dataset.kind===b.dataset.f)?'':'none');}});
</script></body></html>"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--golden", choices=("l1", "phase1"), default="l1",
                   help="l1: evals/datasets/retrieval_golden.jsonl; phase1: eval/golden/golden.jsonl (90 questions)")
    p.add_argument("--split", choices=("dev", "test", "all"), default="test", help="l1 only (phase1 uses all)")
    p.add_argument("--answers", default="0", help="number of L1 questions to answer with the hybrid, or 'all'")
    p.add_argument("--backend", choices=("hcx", "mock"), default=None, help="default: vendor/phase1/configs/model.yaml")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between model calls")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--embed-model", default=None, help="override index.EMBED_MODEL (default: master's KURE-v1)")
    p.add_argument("--dense", choices=("kure", "bge-m3", "ft"), default="kure",
                   help="kure: master KURE-v1; bge-m3: lab V0 dense; ft: bge-m3 + plan 29 weights (--ckpt)")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="ft: a <regime>-e<n>.pt from ft_train.py (default: seen-e2.pt of the latest data/lab/ft run)")
    p.add_argument("--e2e", default="0",
                   help="Phase 1 golden end to end (top 1 table -> summarize_stat -> L3 grade): a number or 'all'")
    p.add_argument("--skip-l1", action="store_true")
    args = p.parse_args()

    if args.embed_model:
        from kosis_rag import index

        index.EMBED_MODEL = args.embed_model
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_golden()
    split, tau = args.split, None
    if args.golden == "phase1":
        use_phase1_golden()
        split, tau = "all", 0.0  # no dev split and no unanswerable rows: the no-answer score does not apply
    ckpt = None
    if args.dense == "ft":
        ckpt = args.ckpt or max((ROOT / "data" / "lab" / "ft").glob("*/seen-e2.pt"), default=None)
        if ckpt is None or not Path(ckpt).exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt} (put the downloaded run folder under data/lab/ft/)")
        ckpt = Path(ckpt).resolve()
        print(f"fine-tuned weights: {ckpt}")
    llm = make_client(args.backend) if args.answers != "0" or args.e2e != "0" else None
    retrievers, dense_label = build_retrievers(args.device, args.dense, ckpt)
    pages = [] if args.skip_l1 else [d / "report.html" for d in run_l1(retrievers, split, tau)]
    if args.answers != "0":
        pages.append(answers(retrievers[f"hybrid-{args.dense}"], llm, args.answers, split, args.interval,
                             dense_label))
    if args.e2e != "0":
        pages.append(e2e(retrievers[f"hybrid-{args.dense}"], args.backend, args.e2e, args.interval, dense_label))
    print("\nopen in a browser:")
    for page in pages:
        print(f"  {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
