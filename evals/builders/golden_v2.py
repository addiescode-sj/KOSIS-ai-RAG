"""Golden set v2 for L1-L4 (T2-32): one question per sampled table, the same qid in every layer.

    python -m evals.builders.golden_v2 draft                    # tables, facts, template questions -> golden_v2_draft.csv
    (write golden_v2_texts.jsonl: one line per qid with the question and L2 measures/dimensions, plus extra items)
    python -m evals.builders.golden_v2 pool                     # check texts, export retrieval_pool_v2.csv to judge
    (fill `slot` in retrieval_pool_v2.csv for every other table that answers the question, `auto`=claude on judged rows)
    python -m evals.builders.golden_v2 finish                   # -> retrieval/query/calculation_golden_v2.jsonl

Tables: every 인구총조사 menu section (parts 3-4 of menu_path) gives one table, every other survey up to five
(seeded), so no survey or table dominates. Facts and L3 refusals come from evals.builders.diverse (Parquet cells,
by code). L2 period, years and regions come from the facts; cycle only when the question names it; measures and
dimensions are the writer's labels (T2-28 vocabulary); a text line may override a code field, e.g. regions
for a 시도 that only the table name holds. Pool candidates that lack a fact's year or a non-total
label are ruled out by code (`auto` column) before anyone judges the rest. Extra items (multi_table, ambiguous,
no_answer) are written by hand in the texts file; a no_answer item lists `absent` terms that no table's
search_text may hold together (with `year` in its periods when given), checked here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from kosis_rag.query import SIDO, SIDO_ORDER

from evals.builders import diverse
from evals.builders.phase1 import sha256
from evals.builders.pool import COLUMNS, POOL_K, apply_labels, assign_splits, pool_rows
from evals.schemas import DATASETS, ROOT, TABLES, CalcItem, QueryItem, RetrievalItem, check_items

WORK = DATASETS / "golden_v2_work.jsonl"
DRAFT = DATASETS / "golden_v2_draft.csv"
TEXTS = DATASETS / "golden_v2_texts.jsonl"
POOL = DATASETS / "retrieval_pool_v2.csv"
OUT_L1 = DATASETS / "retrieval_golden_v2.jsonl"
OUT_L2 = DATASETS / "query_golden_v2.jsonl"
OUT_L3 = DATASETS / "calculation_golden_v2.jsonl"
REVIEW = DATASETS / "golden_v2_review_sample.csv"
PER_SURVEY = 5
# qtype per table in turn, with the L3 item its question is built from (fallback: VAL)
ROTATION = [("exact_name", "VAL"), ("concept", "VAL"), ("multi_condition", "CMP"), ("period_cycle", "TRD"), ("sibling", "VAL")]
OFFICIAL = {name: name for name in SIDO_ORDER} | {alias: name for name, aliases in SIDO.items() for alias in aliases}
L2_CODE_FIELDS = ("period_start", "period_end", "years", "regions", "cycle")
CYCLE_WORDS = {"Y": ("연도별", "매년", "해마다"), "M": ("월별",), "Q": ("분기별",)}


def stratum(info: dict) -> str:
    if info["survey_name"] == "인구총조사":
        return "인구총조사/" + " > ".join(info["menu_path"].split(" > ")[3:5])
    return info["survey_name"]


def sample(rows: list[dict], seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["info"]["coverage"] != "C_수집누락" and diverse.ANNUAL & set(r["prd_se"] or []):
            groups.setdefault(stratum(r["info"]), []).append(r)
    rng = random.Random(seed)
    picked = []
    for name, group in sorted(groups.items()):
        n = 1 if name.startswith("인구총조사/") else PER_SURVEY
        picked += rng.sample(sorted(group, key=lambda r: r["info"]["key"]), min(n, len(group)))
    return picked


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list) -> None:
    path.write_text("".join((r.model_dump_json(exclude_defaults=True) if hasattr(r, "model_dump_json")
                             else json.dumps(r, ensure_ascii=False)) + "\n" for r in rows), encoding="utf-8")


def draft(args) -> int:
    rows = read_jsonl(args.tables)
    work = []
    for n, r in enumerate(sample(rows, args.seed)):
        info = r["info"]
        path = args.values / info["parquet"]
        df, dims = diverse.load(path)
        made = {i.qid.rsplit("-", 1)[1]: i for i in
                diverse.make_items(info["key"], sha256(path), df, dims, random.Random(f"{args.seed}-{info['key']}"), n)}
        if "VAL" not in made:
            continue
        qtype, want = ROTATION[n % len(ROTATION)]
        rep = made.get(want, made["VAL"])
        single = df.item_name.iloc[0] if df.item_id.nunique() == 1 else None
        base = {"survey": info["survey_name"], "table_name": info["table_name"], "single_item": single}
        work.append({**base, "qid": f"V2-{info['key']}", "qtype": qtype, "calc": rep.model_copy(update={"qid": f"V2-{info['key']}"}).model_dump(mode="json", exclude_defaults=True)})
        if n % 5 == 2 and "REF" in made:  # L3-only refusal on the same table, about a fifth of the tables
            work.append({**base, "qid": f"V2-{info['key']}-REF", "qtype": None, "calc": made["REF"].model_copy(update={"qid": f"V2-{info['key']}-REF"}).model_dump(mode="json", exclude_defaults=True)})
    write_jsonl(WORK, work)
    with DRAFT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "qtype", "l3", "survey", "table_name", "item_name", "template", "facts"])
        for x in work:
            c = CalcItem.model_validate(x["calc"])
            facts = "; ".join(f"{f.id}={f.value if f.value is not None else f.raw} {f.period or ''} {f.dims or ''}" for f in c.required_facts + c.aux_facts)
            w.writerow([x["qid"], x["qtype"] or "", f"{c.kind}/{c.type}", x["survey"], x["table_name"], item_name(c, x), c.question, facts])
    tables = Counter(x["calc"]["table_key"] for x in work)
    print(f"{len(work)} items over {len(tables)} tables, {len({x['survey'] for x in work})} surveys -> {DRAFT}")
    return 0


def item_name(calc: CalcItem, work_row: dict) -> str | None:
    _, _, name = diverse.wanted_words(calc)
    return name or work_row.get("single_item")


def l2_from_facts(calc: CalcItem, question: str) -> dict:
    """period_*, years, regions and cycle a parser must read from this question, taken from the facts."""
    facts = [f for f in calc.required_facts + calc.aux_facts if f.source_ref]
    asked = re.findall(r"(\d{4})년(?!생)", question)
    periods = sorted({f.period for f in facts}) if facts else []
    if calc.kind == "not_answerable" and asked:  # no_period asks a year the table lacks
        periods = [asked[0]]
    trend = calc.type == "trend" and len(periods) == 2
    regions = sorted({OFFICIAL[label] for f in facts for label in (f.dims or {}).values() if label in OFFICIAL},
                     key=SIDO_ORDER.index)
    cycle = next((c for c, words in CYCLE_WORDS.items() if any(w in question for w in words)), None)
    return {"period_start": periods[0] if periods else None, "period_end": periods[-1] if periods else None,
            "years": [] if trend else [int(p) for p in periods], "regions": regions, "cycle": cycle}


def label_ok_in_table(label: str, meta: dict) -> bool:
    core = diverse.squash(re.sub(r"\(.*?\)", "", label)) or diverse.squash(label)
    names = [*meta["items"], *(lab for c in meta["classes"] for lab in c["labels"])]
    return any(core in diverse.squash(re.sub(r"\(.*?\)", "", n)) for n in names)


def rule_out(calc: CalcItem, meta: dict) -> str:
    """'' when the table could hold the answer's cells, else why not: a fact's year or a non-total label is missing."""
    for f in calc.required_facts:
        if f.period and f.period not in meta["periods"]:
            return f"rule:no_period {f.period}"
        for name, label in (f.dims or {}).items():
            if not diverse.is_total(label) and not label_ok_in_table(label, meta):
                return f"rule:no_label {label}"
    return ""


def absent_ok(item: dict, metas: dict) -> list[str]:
    """Tables whose search_text holds every `absent` term (and `year` in periods): must be none for no_answer."""
    terms, year = item.get("absent") or [], item.get("year")
    if not terms:
        return ["no `absent` terms given"]
    return [k for k, m in metas.items() if all(t in m["search_text"] for t in terms) and (not year or year in m["periods"])]


def texts_by_qid() -> dict[str, dict]:
    return {t["qid"]: t for t in read_jsonl(TEXTS)}


def final_items(texts: dict, work: list[dict]) -> tuple[list[tuple[dict, CalcItem]], list[str]]:
    out, problems = [], []
    for x in work:
        calc = CalcItem.model_validate(x["calc"])
        t = texts.get(x["qid"])
        if not t:
            problems.append(f"{x['qid']}: no text")
            continue
        # digits of the table name ('만 13세~만 15세이하') may be named too; they come from the registry, not the writer
        template = " ".join([calc.question, *re.findall(r"\d+", x["table_name"])])
        issues = diverse.check_question(calc, template, t["question"], t.get("item_term"), item_name(calc, x))
        if issues:
            problems.append(f"{x['qid']}: {'; '.join(issues)}")
            continue
        out.append((x, calc.model_copy(update={"question": t["question"]})))
    return out, problems


def drafts_for(texts: dict, items: list[tuple[dict, CalcItem]]) -> list[dict]:
    drafts = [{"qid": x["qid"], "split": "test", "qtype": x["qtype"], "source": "v2", "question": c.question}
              for x, c in items if x["qtype"]]
    for t in texts.values():
        if t.get("extra"):
            d = {"qid": t["qid"], "split": "test", "qtype": t["qtype"], "source": "v2", "question": t["question"]}
            if t["qtype"] == "no_answer":
                d |= {"answerable": False, "no_answer_reason": t["no_answer_reason"]}
            drafts.append(d)
    return drafts


def pool(args) -> int:
    from evals.runners.retrieval import bm25_retrievers, load_records

    texts, work = texts_by_qid(), read_jsonl(WORK)
    metas = {r["info"]["key"]: r for r in read_jsonl(args.tables)}
    items, problems = final_items(texts, work)
    for t in texts.values():
        if t.get("extra") and t["qtype"] == "no_answer" and (hits := absent_ok(t, metas)):
            problems.append(f"{t['qid']}: no_answer terms {t.get('absent')} found in {hits[:3]}")
    if problems:
        print("\n".join(problems))
        return 1
    calc_by_qid = {c.qid: c for _, c in items}
    retrievers = bm25_retrievers(load_records(args.tables), POOL_K)
    rows = []
    for d in drafts_for(texts, items):
        found = pool_rows(d, retrievers)
        if d.get("answerable", True) is False:
            rows += [{**r, "auto": ""} for r in found[:3]]
            continue
        gold = [calc_by_qid[d["qid"]].table_key] if d["qid"] in calc_by_qid else [k for s in texts[d["qid"]]["required"] for k in s]
        slot_of = {k: str(i + 1) for i, s in enumerate(texts.get(d["qid"], {}).get("required") or [gold]) for k in s}
        known = {r["key"] for r in found}
        found += [{"qid": d["qid"], "question": d["question"], "qtype": d["qtype"], "rank": "", "found_by": "gold",
                   "key": k, "table_name": metas[k]["info"]["table_name"], "period": metas[k]["info"]["period"], "slot": ""}
                  for k in slot_of if k not in known]
        for r in found:
            if r["key"] in slot_of:
                r |= {"slot": slot_of[r["key"]], "auto": "gold"}
            elif d["qid"] in calc_by_qid:
                r["auto"] = rule_out(calc_by_qid[d["qid"]], metas[r["key"]])
            else:
                r["auto"] = ""
        rows += found
    with POOL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*COLUMNS, "auto"])
        w.writeheader()
        w.writerows(rows)
    todo = sum(1 for r in rows if not r["auto"])
    print(f"{len(rows)} pool rows, {todo} to judge (empty `auto`) -> {POOL}")
    return 0


def finish(args) -> int:
    texts, work = texts_by_qid(), read_jsonl(WORK)
    items, problems = final_items(texts, work)
    if problems:
        print("\n".join(problems))
        return 1
    with POOL.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    l1 = apply_labels(drafts_for(texts, items), rows)
    assign_splits(l1)
    check_items(l1, set(json.loads(line)["info"]["key"] for line in args.tables.read_text(encoding="utf-8").splitlines() if line.strip()))
    split = {i.qid: i.split for i in l1}
    calc_by_qid = {c.qid: c for _, c in items}
    l2 = []
    for i in l1:
        t = texts[i.qid]
        if i.qid in calc_by_qid:
            # the writer may override a code field when the question says more than the cells (a 시도 in a table name)
            fields = l2_from_facts(calc_by_qid[i.qid], i.question) | {k: t[k] for k in L2_CODE_FIELDS if k in t}
        else:
            fields = {k: t.get(k) for k in ("period_start", "period_end", "cycle")} | {"regions": t.get("regions", []), "years": t.get("years", [])}
        l2.append(QueryItem(qid=i.qid, split=i.split, question=i.question, answerable=i.answerable,
                            measures=t["measures"], dimensions=t["dimensions"], **fields))
    l3 = [c.model_copy(update={"split": split[c.qid.removesuffix("-REF")]}) for _, c in items]
    write_jsonl(OUT_L1, l1)
    write_jsonl(OUT_L2, l2)
    write_jsonl(OUT_L3, l3)
    metas = {json.loads(line)["info"]["key"]: json.loads(line)["info"] for line in args.tables.read_text(encoding="utf-8").splitlines() if line.strip()}
    primary = Counter(k for i in l1 for k in i.primary_keys)
    surveys = Counter(metas[k]["survey_name"] for i in l1 for k in i.primary_keys)
    judged = Counter(r["auto"].split(" ")[0] for r in rows)
    info = {
        "l1": {"items": len(l1), "answerable": sum(i.answerable for i in l1), "primary_tables": len(primary),
               "gold_keys": len({k for i in l1 for k in i.gold_keys}), "surveys": dict(surveys.most_common()),
               "qtypes": dict(Counter(i.qtype for i in l1)), "splits": dict(Counter(i.split for i in l1))},
        "l3": {"items": len(l3), "tables": len({c.table_key for c in l3}), "kinds": dict(Counter(f"{c.kind}/{c.type}" for c in l3)),
               "splits": dict(Counter(c.split for c in l3))},
        "pool_rows": dict(judged), "labels": "questions, measures/dimensions and pool judgements by Claude (T2-32); facts, "
                                             "period/years/regions and rule:* pool rows by code",
        "tables.jsonl": sha256(args.tables),
        "sha256": {p.name: sha256(p) for p in (OUT_L1, OUT_L2, OUT_L3, POOL)},
    }
    OUT_L1.with_name("golden_v2.meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "split", "qtype", "question", "required", "measures", "dimensions", "ok", "note"])
        for i in l1[::5]:
            w.writerow([i.qid, i.split, i.qtype, i.question, json.dumps(i.required, ensure_ascii=False),
                        texts[i.qid]["measures"], texts[i.qid]["dimensions"], "", ""])
    print(json.dumps({k: v for k, v in info.items() if k in ("l1", "l3", "pool_rows")}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Golden set v2 for L1-L4 (T2-32)")
    parser.add_argument("step", choices=("draft", "pool", "finish"))
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--values", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args(argv)
    return {"draft": draft, "pool": pool, "finish": finish}[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())
