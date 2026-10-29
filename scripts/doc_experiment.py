"""T2-25: compare search-document variants on a question set with in-memory cosine ranking (same model as Chroma).

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/doc_experiment.py levels --data-dir <provided data>
    PYTHONPATH=... python scripts/doc_experiment.py answers          # answer sets of the proxy questions (no retrieval)
    PYTHONPATH=... python scripts/doc_experiment.py run base v1_region v2_items region+items+head ...
    KOSIS_STAT_GOLDEN=<phase 1 stat_golden.jsonl> adds the 40 phase 1 golden questions to the proxy questions.

Each variant is measured against the base document, not on top of the previous winner, so the order of trials does
not matter. Answer sets come from the original data (raw item names, Parquet region codes, real periods), never from
the cleaned documents, so a variant cannot win by writing the field it is graded on.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from kosis_rag.index import LABEL_CAP, read_records, to_document
from kosis_rag.metadata import REGION_CLASS
from kosis_rag.schemas import TableRecord

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
LEVELS = DATA / "region_levels.json"
OUT = DATA / "doc_experiment"
# Phase 1 golden set (40 questions, one answer table each); not in this repo, so point at it with an env var.
GOLDEN = Path(os.environ.get("KOSIS_STAT_GOLDEN", REPO / "vendor/phase1/eval/golden/stat_golden.jsonl"))
PROXY = REPO / "eval" / "search_proxy_questions.jsonl"

NATION = {"전국", "계", "총계", "합계"}
URBAN = {"동부", "읍부", "면부"}
SIDO = {"서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
        "경기도", "강원도", "강원특별자치도", "충청북도", "충청남도", "전라북도", "전북특별자치도", "전라남도", "경상북도",
        "경상남도", "제주특별자치도", "제주도"}
LEVEL_ORDER = ["전국", "동읍면부", "시도", "시군구", "읍면동", "하위 지역"]


def region_level(code: str, label: str, class_name: str = "") -> str:
    """Level from the label first (works for flat codes), then the KOSIS code length, then the class name."""
    if label in NATION:
        return "전국"
    if label in URBAN:
        return "동읍면부"
    if label in SIDO:
        return "시도"
    n = len(code or "")
    if n == 5:
        return "시군구"
    if 7 <= n <= 10:
        return "읍면동"
    # flat codes (e.g. 법무부 tables): the class name says what the lower level is
    if "읍면동" in class_name:
        return "읍면동"
    if re.search(r"시군구|구시군", class_name):
        return "시군구"
    return "하위 지역"


def is_region_class(levels: dict) -> bool:
    """A class is about regions only if it lists 시도 (국적(지역)별 lists countries and is not one)."""
    return levels.get("시도", 0) >= 3


def build_levels(data_dir: Path) -> None:
    """{table key: {region class name: {level: label count}}} from the Parquet codes."""
    import pyarrow.parquet as pq

    out = {}
    for r in read_records(DATA / "tables.jsonl"):
        path = data_dir / r.info.parquet
        names = pq.read_schema(path).names
        per = {}
        for i in range(1, 9):
            cols = [f"dim{i}_name", f"dim{i}_code", f"dim{i}_label"]
            if cols[0] not in names:
                continue
            for row in pq.read_table(path, columns=cols).group_by(cols).aggregate([]).to_pylist():
                name, code, label = row[cols[0]], row[cols[1]], row[cols[2]]
                if name and label and REGION_CLASS.search(name):
                    lv = per.setdefault(name, {})
                    lv.setdefault(region_level(code, label, name), set()).add(label)
        counts = {n: {k: len(v) for k, v in lv.items()} for n, lv in per.items()}
        out[r.info.key] = {n: c for n, c in counts.items() if is_region_class(c)}
    LEVELS.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} tables -> {LEVELS}")


# ---------- document variants (base = index.to_document) ----------

UNIT_SUFFIX = re.compile(r"\s*\(([^()]*)\)\s*$")


def _clean_item(item: str) -> tuple[str, str | None]:
    m = UNIT_SUFFIX.search(item)
    name, unit = (item[: m.start()], m.group(1).strip()) if m else (item, None)
    name = name.replace("_", " ").replace("ㆍ", "·").replace("–", "-")
    return re.sub(r"\s+", " ", name).strip(), unit


def _region_summary(levels: dict) -> str:
    parts = [f"{lv}({levels[lv]:,})" if lv not in ("전국",) else lv for lv in LEVEL_ORDER if lv in levels]
    return "·".join(parts)


def variant_doc(record: TableRecord, levels: dict, opts: set[str]) -> str:
    """Base document with the chosen cleanups: region (1), items (2), head line (3), paths (4), alias (5)."""
    info = record.info
    items = record.items
    units: list[str] = []
    if "items" in opts:
        cleaned = [_clean_item(i) for i in items]
        items = list(dict.fromkeys(n for n, _ in cleaned))
        units = list(dict.fromkeys(u for _, u in cleaned if u))
    paths = record.menu_paths
    if "paths" in opts:
        words = []
        for p in paths:
            for w in re.split(r"\s*>\s*", p):
                if w and w not in words:
                    words.append(w)
        paths = [" > ".join(words)]
    table_levels = levels.get(info.key, {})
    lines = []
    if "head" in opts:
        reg = " / ".join(_region_summary(v) for v in table_levels.values()) or "지역 구분 없음"
        lines.append(f"{info.table_name} | {'·'.join(record.prd_se)} {record.start}~{record.end} | 지역 단위: {reg} | 항목: {', '.join(items[:8])}")
    lines += [
        f"통계표: {info.table_name}",
        f"조사명: {info.survey_name} | 작성기관: {info.organization}" + (f" | 주제: {record.topic}" if record.topic else ""),
        f"수록기간: {record.start}~{record.end} ({'·'.join(record.prd_se)}), 목록 표기 {info.period}",
        "분류 경로: " + " / ".join(paths),
        "항목: " + ", ".join(items),
    ]
    if units:
        lines.append("단위: " + ", ".join(units))
    for c in record.classes:
        if "region" in opts and c.name in table_levels:
            lines.append(f"{c.name}: 지역 단위 {_region_summary(table_levels[c.name])}")
            continue
        shown = c.labels[:LABEL_CAP]
        more = f" 외 {c.n_labels - len(shown)}개" if c.n_labels > len(shown) else ""
        lines.append(f"{c.name}: " + ", ".join(shown) + more)
    if "alias" in opts:
        alias = [x for x in (record.table_name_en, "통계청" if info.organization == "국가데이터처" else None) if x]
        if alias:
            lines.append("다른 이름: " + " | ".join(alias))
    return "\n".join(lines)


VARIANTS = {
    "base": None,  # index.to_document as loaded in kosis_a_kure_202610291341
    "v1_region": {"region"},
    "v2_items": {"items"},
    "v13_region_head": {"region", "head"},
    "v4_paths": {"paths"},
    "v5_alias": {"alias"},
}


# ---------- questions and answers ----------

def _years(record: TableRecord) -> set[int]:
    return {int(p[:4]) for p in record.periods if p[:4].isdigit()}


def answer_set(cond: dict, records: dict[str, TableRecord], levels: dict) -> set[str]:
    """Tables meeting every condition, judged on the original data only."""
    out = set()
    for key, r in records.items():
        if "survey" in cond and r.info.survey_name != cond["survey"]:
            continue
        if "name_any" in cond and not re.search(cond["name_any"], r.info.table_name):
            continue
        if "item_any" in cond and not any(re.search(cond["item_any"], i) for i in r.items):
            continue
        if "text_any" in cond and not (re.search(cond["text_any"], r.info.table_name)
                                       or any(re.search(cond["text_any"], i) for i in r.items)):
            continue
        if "class_any" in cond and not any(re.search(cond["class_any"], c.name) for c in r.classes):
            continue
        if "region" in cond and not any(cond["region"] in lv for lv in levels.get(key, {}).values()):
            continue
        if "year" in cond and cond["year"] not in _years(r):
            continue
        if "cycle" in cond and cond["cycle"] not in set(r.cycle) | set(r.prd_se):
            continue
        out.add(key)
    return out


def load_questions(records, levels) -> list[dict]:
    qs = []
    if not GOLDEN.exists():
        print(f"golden set not found at {GOLDEN} (set KOSIS_STAT_GOLDEN); proxy questions only", file=sys.stderr)
    for g in map(json.loads, GOLDEN.open(encoding="utf-8")) if GOLDEN.exists() else []:
        qs.append({"id": g["id"], "q": g["question"], "set": "golden40", "answers": {Path(g["table"]).stem}})
    for p in map(json.loads, PROXY.open(encoding="utf-8")):
        qs.append({"id": p["id"], "q": p["q"], "set": "proxy30", "answers": answer_set(p["cond"], records, levels)})
    return qs


# ---------- embedding and scoring ----------

def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("nlpai-lab/KURE-v1")


def embed(model, texts: list[str]) -> np.ndarray:
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = np.zeros((len(texts), 1024), dtype=np.float32)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        vecs[idx] = model.encode([texts[i] for i in idx], batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
    return vecs


def ranks(doc_vecs: np.ndarray, keys: list[str], q_vecs: np.ndarray, qs: list[dict]) -> list[int | None]:
    """1-based rank of the first answer table per question (None if no answer in the set)."""
    sims = q_vecs @ doc_vecs.T
    out = []
    for i, q in enumerate(qs):
        order = np.argsort(-sims[i])
        pos = [n for n, j in enumerate(order, 1) if keys[j] in q["answers"]]
        out.append(pos[0] if pos else None)
    return out


def metrics(rk: list[int | None]) -> dict:
    n = len(rk)
    hit = lambda k: sum(1 for r in rk if r and r <= k)
    return {"n": n, **{f"hit@{k}": hit(k) for k in (1, 5, 10, 20)},
            "mrr": round(sum(1 / r for r in rk if r) / n, 4)}


def run(names: list[str]) -> None:
    records = {r.info.key: r for r in read_records(DATA / "tables.jsonl")}
    levels = json.loads(LEVELS.read_text(encoding="utf-8"))
    qs = load_questions(records, levels)
    keys = list(records)
    OUT.mkdir(exist_ok=True)
    model = _model()
    q_vecs = embed(model, [q["q"] for q in qs])
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    for name in names:
        opts = set(name.split("+")) if "+" in name else VARIANTS.get(name)
        docs = [to_document(records[k]) if opts is None else variant_doc(records[k], levels, opts) for k in keys]
        cache = OUT / f"{name}.npy"
        t0 = time.perf_counter()
        if cache.exists():
            vecs = np.load(cache)
        else:
            vecs = embed(model, docs)
            np.save(cache, vecs)
        secs = time.perf_counter() - t0
        rk = ranks(vecs, keys, q_vecs, qs)
        res = {"all": metrics(rk),
               "golden40": metrics([r for r, q in zip(rk, qs) if q["set"] == "golden40"]),
               "proxy30": metrics([r for r, q in zip(rk, qs) if q["set"] == "proxy30"]),
               "doc_chars_mean": round(sum(map(len, docs)) / len(docs)), "doc_chars_max": max(map(len, docs)),
               "embed_s": round(secs), "ranks": {q["id"]: r for q, r in zip(qs, rk)}}
        summary[name] = res
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, json.dumps({k: v for k, v in res.items() if k != "ranks"}, ensure_ascii=False), flush=True)


def show_answers() -> None:
    records = {r.info.key: r for r in read_records(DATA / "tables.jsonl")}
    levels = json.loads(LEVELS.read_text(encoding="utf-8"))
    for q in load_questions(records, levels):
        if q["set"] == "proxy30":
            sample = sorted(q["answers"])[:3]
            print(f"{q['id']} | {q['q']} | {len(q['answers'])} tables | e.g. {[records[k].info.table_name[:24] for k in sample]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["levels", "answers", "run"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "levels":
        build_levels(args.data_dir)
    elif args.cmd == "answers":
        show_answers()
    else:
        run(args.names or list(VARIANTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
