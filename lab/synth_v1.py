"""Training questions v1 for the n3 retraining (plan 47): slots first, sentences after, gold as a set of tables.

synth_v0 (plan 17) gave each question one gold table, 95% of questions a year, and fixed 3 questions per table. Here a
question starts from a slot spec drawn from one anchor table (what is counted, breakdowns, region, period, intent), and
its gold set is every table the spec fits, computed by the same rules for all 2,364 tables (no table-specific fixes):

    positives  measure key shared, every asked breakdown held, region / period / cycle covered
    no_neg     not a positive but too close to use as a negative (asked label missing from a held breakdown, period
               only partly covered, or a measure that contains / is contained in the asked one)
    hard_neg   same measure (or same table name) but a slot the table does not hold: the tables search must learn
               to tell apart

Evaluation tables are never used, anywhere: routing_v0 gold + alt, retrieval_golden required groups, the Phase 1
golden tables and the human e2e golden tables. A spec whose gold set touches one is dropped; they are also removed from
no_neg and from every negative list.

Stages (data/ is gitignored; nothing here reads a golden question, only golden table keys):
    python lab/synth_v1.py plan                  slot specs -> data/lab/synth/v1/specs.jsonl (+ claude / hcx cards)
    python lab/synth_v1.py rule                  keyword-style questions for the rule share -> v1/rule.jsonl
    python lab/synth_v1_hcx.py                   HCX share (separate script, calls the API)
    python lab/synth_v1.py build                 specs + sentences -> data/lab/synth/synth_v1.jsonl + report
    python lab/synth_v1.py --selftest            rule checks on a few fake tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "vendor" / "phase1" / "src", ROOT / "vendor" / "phase1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TABLES = ROOT / "data" / "tables.jsonl"
OUT = ROOT / "data" / "lab" / "synth" / "v1"
FINAL = ROOT / "data" / "lab" / "synth" / "synth_v1.jsonl"
GOLDEN = ROOT / "eval" / "golden"
VERSION = "synth-v1"

# Phase 1 golden tables (golden.jsonl, 90 questions over these 10 tables); also read from the file when present
PHASE1_TABLES = ["101_DT_1B8000H", "101_DT_1B08024", "101_DT_1JC1502", "101_DT_101094A001", "101_DT_1B8000F",
                 "101_DT_1B83A35", "101_DT_1BPA002", "101_DT_1BPB301", "110_TX_11025_A000_A", "110_DT_110025_A038_A"]

# plan 47 §4.3 defaults
TARGET = 10_000
MIN_PER_TABLE = 2
SHARES_GEN = {"claude": 0.5, "hcx": 0.3, "rule": 0.2}
SHARES_SPEC = {"specific": 0.70, "broad": 0.15, "concept": 0.15}
SHARES_INTENT = {"value": 0.30, "trend": 0.20, "compare": 0.15, "composition": 0.15, "rank": 0.10, "lookup": 0.10}
SHARES_PERIOD = {"year": 0.55, "none": 0.30, "span": 0.15}  # span: range, recent, month, quarter, cycle words
RERANK_MAX_POS = 20  # bigger gold sets are kept for the embedding only (a re-ranker group needs real negatives)
TOTAL_LABELS = {"계", "합계", "총계", "소계", "전체", "전국", "총", "계(전체)", "전국(계)", "총인구"}
REGION_KEYS = {"시도", "시군구", "읍면동", "행정구역"}
LATEST = 2025  # the newest year in the provided data (recent = this or the year before)

# Concepts: a phenomenon people name instead of the indicator (plan 47 §4.5). Written from the KOSIS population
# indicator system, not from golden questions. A table fits when one of its measure keys contains a listed word.
CONCEPTS: dict[str, list[str]] = {
    "고령화": ["고령자", "노령화", "고령인구", "노년부양비", "중위연령", "평균연령"],
    "저출산": ["출생", "출산"],
    "결혼 감소": ["혼인"],
    "이혼 증가": ["이혼"],
    "사망 추세": ["사망"],
    "인구 자연감소": ["자연증가", "출생", "사망"],
    "1인 가구 증가": ["1인가구"],
    "외국인 증가": ["외국인", "결혼이민", "귀화"],
    "다문화": ["다문화"],
    "인구 이동": ["이동인구", "전입", "전출", "순이동"],
    "출퇴근": ["통근", "통학"],
    "일자리": ["취업", "유업자", "경제활동"],
    "주거 형태": ["주택", "거처"],
    "경력 단절": ["경력단절"],
    "활동 제약": ["활동제약"],
    "인구 규모": ["총인구", "인구수"],
    "장래 인구": ["추계인구", "장래인구"],
    "가구 구성": ["세대구성", "가구원"],
}

# Everyday words for the rule share (global statistics terms; no golden question was read to write it)
EVERYDAY = {
    "출생아수": "태어난 아이 수", "출생아": "신생아", "사망자수": "사망자", "혼인건수": "결혼 건수", "혼인": "결혼",
    "이혼건수": "이혼 건수", "취업자": "일하는 사람", "취업인구": "일하는 사람", "이동인구": "이사한 사람",
    "통근통학인구": "출퇴근 통학 인구", "통근·통학인구": "출퇴근 통학 인구", "가구주": "세대주", "고령자": "노인",
    "1인가구": "혼자 사는 가구", "외국인": "외국 사람", "가구": "집", "기혼여성인구": "결혼한 여성",
    "유업자": "일하는 사람", "경제활동인구": "경제활동 인구", "주택": "집",
}
SIDO_SHORT = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주",
              "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원도": "강원",
              "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북",
              "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주"}


def compact(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


# ---------------------------------------------------------------- table profiles

@dataclass
class Profile:
    key: str
    name: str  # normalized table name (kosis_rag.text.norm_name)
    raw_name: str
    survey: str
    measure: str | None
    mkeys: list[str]
    dims: set[str]  # breakdown keys the table holds (classes + implied), region keys included
    labels: dict[str, list[str]]  # breakdown key -> labels without totals
    region_level: set[str]
    regions: list[str] | None  # official 시도 names
    sigungu: list[str]  # 시군구-looking labels of region classes
    years: list[int]
    cycle: set[str]
    months: list[str]  # YYYYMM periods
    quarters: list[str]
    items: list[str]


def load_profiles(path: Path = TABLES) -> dict[str, Profile]:
    from kosis_rag.fields import table_fields
    from kosis_rag.schemas import TableRecord
    from kosis_rag.text import norm_item, norm_label, norm_name
    from kosis_rag.vocab import SIDO_LABELS, dimension_keys, normalize_dimension

    out: dict[str, Profile] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = TableRecord.model_validate_json(line)
            tf = table_fields(rec)
            labels: dict[str, list[str]] = {}
            sigungu: list[str] = []
            for c in rec.classes:
                keys = dimension_keys(c.name) or [normalize_dimension(c.name)]
                labs = [norm_label(x) for x in c.labels if x.strip() and compact(x) not in TOTAL_LABELS]
                for k in keys:
                    labels.setdefault(k, [])
                    labels[k] += [x for x in labs if x not in labels[k]]
                if set(keys) & REGION_KEYS or any(x in SIDO_LABELS for x in c.labels):
                    sigungu += [x for x in labs if x not in SIDO_LABELS and re.search(r"[가-힣](시|군|구)$", x)]
            periods = [str(p) for p in rec.periods]
            out[rec.info.key] = Profile(
                key=rec.info.key, name=norm_name(rec.info.table_name)[0], raw_name=rec.info.table_name,
                survey=rec.info.survey_name, measure=tf.measure, mkeys=list(tf.measure_keys),
                dims=set(tf.dimension_keys) | set(tf.implied_keys) | set(tf.region_level),
                labels=labels, region_level=set(tf.region_level), regions=tf.regions,
                sigungu=list(dict.fromkeys(sigungu)), years=tf.available_years, cycle=set(tf.cycle),
                months=[p for p in periods if re.fullmatch(r"\d{6}", p) and int(p[4:]) <= 12],
                quarters=[p for p in periods if re.fullmatch(r"\d{4}0?[1-4]Q|\d{4}Q[1-4]", p)],
                items=[i for i in dict.fromkeys(norm_item(x) for x in rec.items) if i and compact(i) not in TOTAL_LABELS],
            )
    return out


# ---------------------------------------------------------------- evaluation tables (never used)

def eval_tables(golden: Path = GOLDEN) -> dict[str, set[str]]:
    """Every table any evaluation set accepts, by source. Missing files are skipped (and reported)."""
    src: dict[str, set[str]] = {"phase1": set(PHASE1_TABLES)}
    p = golden / "routing_v0.jsonl"
    if p.exists():
        src["routing_v0"] = {k for r in _jsonl(p) for k in [r["table"], *r.get("alt_tables", [])]}
    p = golden / "retrieval_golden.jsonl"
    if p.exists():
        src["retrieval_golden"] = {k for r in _jsonl(p) for g in r.get("required") or [] for k in g}
    for name in ("golden.jsonl", "stat_golden.jsonl", "m1_golden.jsonl"):
        p = golden / name
        if p.exists():
            for r in _jsonl(p):
                for k in ("table_id", "table", "table_key"):
                    if isinstance(r.get(k), str):
                        src["phase1"].add(r[k] if "_" in r[k][:4] else r[k])
    p = golden / "human_e2e" / "questions.csv"
    if p.exists():
        with p.open(encoding="utf-8-sig") as f:
            src["human_e2e"] = {k.strip() for r in csv.DictReader(f) for k in re.split(r"[;|,\s]+", r["table_keys"] or "")
                                if k.strip()}
    return src


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------------------------------------------------------- slot specs and gold sets

@dataclass
class Spec:
    id: str
    anchor: str
    specificity: str
    intent: str
    measure: str | None  # what the question counts, as the anchor table words it
    mkeys: list[str]  # measure keys a positive must share (concept: the concept's measure words, matched by contains)
    concept: str | None = None
    dims: list[str] = field(default_factory=list)  # breakdowns asked ("시군구별")
    labels: list[list[str]] = field(default_factory=list)  # [breakdown key, label] values asked
    region_level: str | None = None
    regions: list[str] = field(default_factory=list)  # official 시도 names (compare may ask two)
    sigungu: list[str] = field(default_factory=list)
    period: dict = field(default_factory=dict)  # {"type": year|range|recent|month|quarter|none, ...}
    cycle: str | None = None  # M / Q asked by words (월별, 분기별)
    item: str | None = None  # an item name, wording only (not a gold-set condition)
    generator: str = "claude"
    style: str = ""

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def measure_fit(p: Profile, spec: Spec, anchor: Profile) -> str:
    """'same' | 'near' (one measure key contains the other) | 'no'."""
    if spec.concept:
        return "same" if any(w in mk for mk in p.mkeys for w in spec.mkeys) else "no"
    if not spec.mkeys:  # the anchor's name does not say what it counts: same table name only
        return "same" if p.name == anchor.name else "no"
    if set(p.mkeys) & set(spec.mkeys):
        return "same"
    if any(a in b or b in a for a in p.mkeys for b in spec.mkeys if min(len(a), len(b)) >= 2):
        return "near"
    return "no"


def judge(p: Profile, spec: Spec, anchor: Profile) -> tuple[str, list[str]]:
    """('pos' | 'soft' | 'neg' | 'other', reasons). soft = too close to be a negative; other = unrelated."""
    fit = measure_fit(p, spec, anchor)
    if fit == "no":
        return ("neg", ["same name"]) if p.name == anchor.name and p.key != anchor.key else ("other", [])
    hard, soft = [], []
    for d in spec.dims:
        if d not in p.dims:
            hard.append(f"no {d}")
    for d, lab in spec.labels:
        if d not in p.dims:
            hard.append(f"no {d}")
        elif d in p.labels and p.labels[d] and compact(lab) not in {compact(x) for x in p.labels[d]}:
            soft.append(f"{d} lacks {lab}")
    if spec.region_level and spec.region_level not in p.region_level:
        hard.append(f"no {spec.region_level} level")
    for r in spec.regions:
        if p.regions is None or r not in p.regions:
            hard.append(f"no {r}")
    for s in spec.sigungu:
        if "시군구" not in p.region_level:
            hard.append("no 시군구")
        elif compact(s) not in {compact(x) for x in p.sigungu}:
            soft.append(f"lacks {s}")
    if spec.cycle and spec.cycle not in p.cycle:
        hard.append(f"cycle {spec.cycle}")
    per, years = spec.period, set(p.years)
    if per.get("type") == "year":
        if not set(per["years"]) <= years:
            (soft if set(per["years"]) & years else hard).append("year")
    elif per.get("type") == "range":
        span = set(range(per["from"], per["to"] + 1))
        if not ({per["from"], per["to"]} <= years):
            (soft if span & years else hard).append("range")
    elif per.get("type") == "recent":
        if not years or max(years) < LATEST - 1:
            (soft if years and max(years) >= LATEST - 5 else hard).append("not recent")
    elif per.get("type") == "month":
        if per["ym"] not in p.months:
            (soft if "M" in p.cycle else hard).append("month")
    elif per.get("type") == "quarter":
        if "Q" not in p.cycle:
            hard.append("quarter")
    if hard:
        return "neg", hard
    if soft or fit == "near":
        return "soft", soft or ["near measure"]
    return "pos", []


def gold_sets(spec: Spec, profiles: dict[str, Profile], index: dict[str, set[str]]) -> dict[str, list]:
    anchor = profiles[spec.anchor]
    cand = set(index.get(f"name:{anchor.name}", set()))
    if spec.concept:
        for mk, keys in index.items():
            if not mk.startswith("name:") and any(w in mk for w in spec.mkeys):
                cand |= keys
    else:
        for mk in spec.mkeys:
            cand |= index.get(mk, set())
        for mk, keys in index.items():  # near measures (contains / contained), for no_neg
            if not mk.startswith("name:") and any(min(len(mk), len(a)) >= 2 and (mk in a or a in mk) for a in spec.mkeys):
                cand |= keys
    pos, soft, neg = [], [], []
    for k in sorted(cand):
        verdict, why = judge(profiles[k], spec, anchor)
        if verdict == "pos":
            pos.append(k)
        elif verdict == "soft":
            soft.append(k)
        elif verdict == "neg":
            neg.append((k, why))
    return {"pos": pos, "soft": soft, "neg": neg}


def build_index(profiles: dict[str, Profile]) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for p in profiles.values():
        for mk in p.mkeys:
            idx[mk].add(p.key)
        idx[f"name:{p.name}"].add(p.key)
    return idx


def world_index(profiles: dict[str, Profile], banned: set[str], strict: bool) -> dict[str, set[str]]:
    """The tables a gold set may draw from. Default: evaluation tables do not exist for training at all, so they are
    never gold, never no_neg, never a negative, and a question near them is judged among the other tables only.
    strict: they stay in the index and any spec whose gold set holds one is dropped (fewer, narrower questions)."""
    return build_index(profiles if strict else {k: p for k, p in profiles.items() if k not in banned})


def _pick(rng: random.Random, shares: dict[str, float]) -> str:
    return rng.choices(list(shares), weights=list(shares.values()))[0]


def concept_of(p: Profile) -> str | None:
    hits = [c for c, words in CONCEPTS.items() if any(w in mk for mk in p.mkeys for w in words)]
    return hits[0] if hits else None


def _label_like(item: str, p: Profile) -> bool:
    """Items that are really breakdown values (a region, a year, a bare 남자/여자) say nothing about what is counted."""
    c = compact(item)
    if c in {"남자", "여자", "남", "여", "남성", "여성", "계", "기타", "평균", "비율", "구성비"}:
        return True
    if c in {compact(x) for x in SIDO_SHORT} | set(SIDO_SHORT.values()) or re.fullmatch(r"[\d.~\-년세]+", c):
        return True
    return any(c in {compact(x) for x in labs} for labs in p.labels.values())


def draw_spec(rng: random.Random, p: Profile, sid: str, profiles, index, banned: set[str]) -> Spec | None:
    """One slot spec from table p, or None when the draw cannot avoid the evaluation tables."""
    specificity = _pick(rng, SHARES_SPEC)
    concept = concept_of(p) if specificity == "concept" else None
    if specificity == "concept" and not concept:
        specificity = "specific"
    intent = _pick(rng, SHARES_INTENT)
    if intent == "trend" and len(p.years) < 3 and rng.random() < 0.7:
        intent = "value"  # 1,926 tables hold one year (census tables split by year); a trend needs several
    nonreg = [d for d in p.labels if d not in REGION_KEYS and d in p.dims and p.labels[d]]
    if intent == "rank" and not (p.region_level or nonreg):
        intent = "value"
    if intent == "compare" and not ((p.regions and len(p.regions) >= 2) or any(len(p.labels[d]) >= 2 for d in nonreg)):
        intent = "value"
    if intent == "composition" and not nonreg:
        intent = "value"
    s = Spec(id=sid, anchor=p.key, specificity=specificity, intent=intent, measure=p.measure,
             mkeys=list(CONCEPTS[concept]) if concept else list(p.mkeys), concept=concept)

    # period
    ptype = _pick(rng, SHARES_PERIOD)
    if intent == "trend" and ptype == "year":
        ptype = "span"
    if not p.years:
        ptype = "none"
    if ptype == "year":
        s.period = {"type": "year", "years": [rng.choice(p.years)]}
    elif ptype == "span":
        opts = ["range"] if len(p.years) >= 3 else []
        opts += ["recent"] if max(p.years) >= LATEST - 1 else []
        opts += ["month"] if p.months and intent != "trend" else []
        opts += ["quarter"] if "Q" in p.cycle else []
        opts += ["cycle"] if p.cycle & {"M", "Q"} else []
        kind = rng.choice(opts) if opts else ("none" if intent == "trend" else "year")
        if kind == "range":
            a, b = sorted(rng.sample(p.years, 2))
            while b - a < 2 and len(p.years) >= 3:
                a, b = sorted(rng.sample(p.years, 2))
            s.period = {"type": "range", "from": a, "to": b}
        elif kind == "recent":
            s.period = {"type": "recent"}
        elif kind == "month":
            ym = rng.choice(p.months)
            s.period = {"type": "month", "ym": ym}
            s.cycle = "M"
        elif kind == "quarter":
            s.period = {"type": "quarter", "year": rng.choice(p.years)}
            s.cycle = "Q"
        elif kind == "year":
            s.period = {"type": "year", "years": [rng.choice(p.years)]}
        elif kind == "cycle":
            s.cycle = rng.choice(sorted(p.cycle & {"M", "Q"}))
            s.period = {"type": "none"}
        else:
            s.period = {"type": "none"}
    else:
        s.period = {"type": "none"}
    if s.period.get("type") == "year" and intent == "compare" and len(p.years) >= 2 and rng.random() < 0.3:
        s.period = {"type": "year", "years": sorted(rng.sample(p.years, 2))}  # compare two years

    # intent slots
    if intent == "rank":
        if p.region_level and (rng.random() < 0.7 or not nonreg):
            s.region_level = rng.choice(sorted(p.region_level))
        else:
            s.dims.append(rng.choice(nonreg))
    elif intent == "compare":
        two = [d for d in nonreg if len(p.labels[d]) >= 2]
        if p.regions and len(p.regions) >= 2 and (rng.random() < 0.6 or not two):
            s.regions = sorted(rng.sample(p.regions, 2))
        else:
            d = rng.choice(two)
            for lab in rng.sample(p.labels[d], min(2, len(p.labels[d]))):
                s.labels.append([d, lab])
    elif intent == "composition":
        s.dims.append(rng.choice(nonreg))
    elif intent in ("value", "trend") and specificity != "broad":
        if p.regions and rng.random() < 0.5:
            if p.sigungu and "시군구" in p.region_level and rng.random() < 0.3:
                s.sigungu = [rng.choice(p.sigungu)]
            else:
                s.regions = [rng.choice(p.regions)]
    items = [i for i in p.items[:40] if not _label_like(i, p)]
    if specificity != "concept" and len(items) > 1 and rng.random() < 0.3:
        s.item = rng.choice(items)

    # specific: add the anchor's own slots that narrow the gold set, a few at most (sibling tables get told apart)
    sets = gold_sets(s, profiles, index)
    if specificity == "specific":
        target = rng.choice([1, 2, 3, 3, 5, 5, 8])
        for _ in range(3):
            if len(sets["pos"]) <= target:
                break
            options = []
            for d in nonreg:
                if d not in s.dims and d not in {x for x, _ in s.labels}:
                    options.append(("dim", d))
                    if p.labels[d]:
                        options.append(("label", d, rng.choice(p.labels[d])))
            for lv in sorted(p.region_level - {s.region_level}):
                if not s.region_level:
                    options.append(("level", lv))
            if p.cycle & {"M", "Q"} and not s.cycle:
                options.append(("cycle", rng.choice(sorted(p.cycle & {"M", "Q"}))))
            if not options:
                break
            scored = []
            for opt in rng.sample(options, min(8, len(options))):
                t = Spec(**{**s.__dict__, "dims": list(s.dims), "labels": [list(x) for x in s.labels]})
                if opt[0] == "dim":
                    t.dims.append(opt[1])
                elif opt[0] == "label":
                    t.labels.append([opt[1], opt[2]])
                elif opt[0] == "level":
                    t.region_level = opt[1]
                else:
                    t.cycle = opt[1]
                ts = gold_sets(t, profiles, index)
                if p.key in ts["pos"]:
                    scored.append((len(ts["pos"]), rng.random(), t, ts))
            if not scored:
                break
            scored.sort(key=lambda x: (x[0], x[1]))
            if scored[0][0] >= len(sets["pos"]):
                break
            _, _, s, sets = scored[0]
    if p.key not in sets["pos"]:
        return None  # the slots contradict the anchor (should not happen; kept as a guard)
    if banned & set(sets["pos"]):  # only with --strict-eval (else the index never holds an evaluation table)
        return None
    return s


MAX_PER_TABLE = 12


def allocate(rng: random.Random, profiles: dict[str, Profile], usable: list[str], total: int) -> Counter:
    """Questions per table: MIN_PER_TABLE each; the rest split by survey (sqrt of its table count), then inside a
    survey by measure group (sqrt of the group size), at most MAX_PER_TABLE per table. Census tables are 89% of the
    usable tables, so a flat split would make them most of the data again (plan 47 D5)."""
    surveys: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for k in usable:
        p = profiles[k]
        surveys[p.survey][p.mkeys[0] if p.mkeys else f"name:{p.name}"].append(k)
    alloc = Counter({k: MIN_PER_TABLE for k in usable})
    extra = max(0, total - sum(alloc.values()))
    s_w = {sv: math.sqrt(sum(len(ks) for ks in g.values())) for sv, g in surveys.items()}
    for sv, groups in surveys.items():
        s_extra = extra * s_w[sv] / sum(s_w.values())
        g_w = {g: math.sqrt(len(ks)) for g, ks in groups.items()}
        for g, ks in groups.items():
            n = round(s_extra * g_w[g] / sum(g_w.values()))
            order = rng.sample(ks, len(ks))
            room = sum(MAX_PER_TABLE - alloc[k] for k in ks)
            for i in range(min(n, room)):
                k = order[i % len(order)]
                while alloc[k] >= MAX_PER_TABLE:
                    i += 1
                    k = order[i % len(order)]
                alloc[k] += 1
    short = total - sum(alloc.values())  # what the caps left over goes to the other tables, evenly
    order = rng.sample(usable, len(usable))
    i = 0
    while short > 0 and i < len(order) * MAX_PER_TABLE:
        k = order[i % len(order)]
        if alloc[k] < MAX_PER_TABLE:
            alloc[k] += 1
            short -= 1
        i += 1
    return alloc


def assign_generators(rng: random.Random, specs: list[Spec]) -> None:
    for s in specs:
        g = _pick(rng, SHARES_GEN)
        if g == "rule" and s.specificity == "concept":
            g = rng.choice(["claude", "hcx"])  # a keyword string cannot hide the indicator name
        s.generator = g
        s.style = rng.choice(["구어체 질문", "짧은 질문", "격식체 질문", "상황 설명 + 질문", "요청형(보여줘·알려줘)"]) \
            if g != "rule" else "검색어"


def card(s: Spec, profiles: dict[str, Profile]) -> str:
    """The slot spec in words, for the writers (Claude, HCX). No table name: the writer must not copy it."""
    parts = [f"의도: {INTENT_KO[s.intent]}"]
    if s.concept:
        parts.append(f"개념: {s.concept} (지표 이름을 직접 말하지 말고 현상으로)")
    else:
        parts.append(f"세는 대상: {s.measure or profiles[s.anchor].name}")
    if s.specificity == "broad":
        parts.append("구체성: 넓게(조건 거의 없이)")
    if s.item:
        parts.append(f"항목: {s.item}")
    if s.dims:
        parts.append("분류: " + ", ".join(f"{d}별" for d in s.dims))
    for d, lab in s.labels:
        parts.append(f"{d}={lab}")
    if s.region_level:
        parts.append(f"지역 단위: {s.region_level}별")
    if s.regions:
        parts.append("지역: " + ", ".join(s.regions))
    if s.sigungu:
        parts.append("시군구: " + ", ".join(s.sigungu))
    parts.append("기간: " + period_ko(s))
    if s.cycle:
        parts.append(f"주기: {'월별' if s.cycle == 'M' else '분기별'}")
    parts.append(f"말투: {s.style}")
    return " | ".join(parts)


INTENT_KO = {"value": "값 조회", "trend": "추이·변화", "compare": "비교", "composition": "구성·분포", "rank": "순위·가장 많은 곳",
             "lookup": "자료·통계표 찾기"}


def period_ko(s: Spec) -> str:
    per = s.period
    t = per.get("type")
    if t == "year":
        return "·".join(f"{y}년" for y in per["years"])
    if t == "range":
        return f"{per['from']}~{per['to']}년"
    if t == "recent":
        return "최근(연도 숫자 쓰지 말 것)"
    if t == "month":
        return f"{per['ym'][:4]}년 {int(per['ym'][4:])}월"
    if t == "quarter":
        return f"{per['year']}년 분기"
    return "언급 없음(연도 쓰지 말 것)"


def cmd_plan(args) -> int:
    rng = random.Random(args.seed)
    profiles = load_profiles(args.tables)
    src = eval_tables(args.golden)
    banned = set().union(*src.values())
    usable = sorted(k for k in profiles if k not in banned)
    index = world_index(profiles, banned, args.strict_eval)
    alloc = allocate(rng, profiles, usable, args.target)
    specs, fails = [], Counter()
    n = 0
    for k in usable:
        got = 0
        for _ in range(alloc[k] * 4):  # retries when a draw touches an evaluation table
            if got >= alloc[k]:
                break
            n += 1
            s = draw_spec(rng, profiles[k], f"V-{n:06d}", profiles, index, banned)
            if s is None:
                fails[k] += 1
                continue
            specs.append(s)
            got += 1
    # renumber in order, dedupe identical slot sets
    seen, uniq = set(), []
    for s in specs:
        sig = json.dumps({k: v for k, v in s.to_json().items() if k not in ("id",)}, ensure_ascii=False, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            uniq.append(s)
    for i, s in enumerate(uniq, 1):
        s.id = f"V-{i:06d}"
    assign_generators(rng, uniq)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "specs.jsonl").open("w", encoding="utf-8") as f:
        for s in uniq:
            f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
    for g in ("claude", "hcx"):
        with (OUT / f"cards_{g}.jsonl").open("w", encoding="utf-8") as f:
            for s in uniq:
                if s.generator == g:
                    f.write(json.dumps({"id": s.id, "card": card(s, profiles)}, ensure_ascii=False) + "\n")
    meta = {"version": VERSION, "seed": args.seed, "target": args.target, "tables": len(profiles),
            "eval_tables": {k: len(v) for k, v in src.items()}, "eval_union": len(banned), "usable_tables": len(usable),
            "specs": len(uniq), "tables_with_specs": len({s.anchor for s in uniq}),
            "tables_short": sum(1 for k in usable if fails[k] and sum(1 for s in uniq if s.anchor == k) < alloc[k]),
            "generator": Counter(s.generator for s in uniq), "specificity": Counter(s.specificity for s in uniq),
            "intent": Counter(s.intent for s in uniq), "period": Counter(s.period.get("type") for s in uniq)}
    (OUT / "plan_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- rule share (keyword style)

def rule_question(rng: random.Random, s: Spec, p: Profile) -> str:
    what = s.measure or p.name
    if rng.random() < 0.35:
        for a, b in sorted(EVERYDAY.items(), key=lambda x: -len(x[0])):
            if compact(what).startswith(a):
                what = b + compact(what)[len(a):]
                break
    bits = [what] + ([s.item] if s.item and compact(s.item) not in compact(what) else [])
    per = s.period
    if per.get("type") == "year":
        bits += [f"{y}" + ("년" if rng.random() < 0.6 else "") for y in per["years"]]
    elif per.get("type") == "range":
        bits.append(f"{per['from']}~{per['to']}")
    elif per.get("type") == "recent":
        bits.append(rng.choice(["최근", "최근 5년", "요즘"]))
    elif per.get("type") == "month":
        bits.append(f"{per['ym'][:4]}년 {int(per['ym'][4:])}월")
    elif per.get("type") == "quarter":
        bits.append(f"{per['year']} 분기별")
    if s.cycle and per.get("type") not in ("month", "quarter"):
        bits.append("월별" if s.cycle == "M" else "분기별")
    for r in s.regions:
        bits.append(SIDO_SHORT.get(r, r) if rng.random() < 0.7 else r)
    bits += s.sigungu
    if s.region_level:
        bits.append(f"{s.region_level}별")
    bits += [f"{d}별" for d in s.dims]
    bits += [lab for _, lab in s.labels]
    if s.intent == "trend":
        bits.append(rng.choice(["추이", "변화", ""]))
    elif s.intent == "rank":
        bits.append(rng.choice(["순위", "가장 많은 곳", ""]))
    elif s.intent == "lookup":
        bits.append(rng.choice(["통계", "자료", "통계표"]))
    bits = [b for b in bits if b]
    head, rest = bits[0], bits[1:]
    rng.shuffle(rest)
    cut = rng.randint(0, len(rest))
    q = " ".join(dict.fromkeys(" ".join(rest[:cut] + [head] + rest[cut:]).split()))
    if rng.random() < 0.15:
        q = q.replace(" ", "")
    return q


def cmd_rule(args) -> int:
    rng = random.Random(args.seed + 1)
    profiles = load_profiles(args.tables)
    specs = _jsonl(OUT / "specs.jsonl")
    n = 0
    with (OUT / "rule.jsonl").open("w", encoding="utf-8") as f:
        for d in specs:
            if d["generator"] != "rule":
                continue
            s = Spec(**d)
            f.write(json.dumps({"id": s.id, "question": rule_question(rng, s, profiles[s.anchor])}, ensure_ascii=False) + "\n")
            n += 1
    print(f"rule questions: {n} -> {OUT / 'rule.jsonl'}")
    return 0


# ---------------------------------------------------------------- build: specs + sentences -> training rows

YEAR_RE = re.compile(r"(?<!\d)(19|20)\d\d(?!\d)")


def surface_check(q: str, s: Spec, p: Profile) -> list[str]:
    """Problems a sentence has against its spec (empty = fine)."""
    bad = []
    t = s.period.get("type")
    if t == "year":
        for y in s.period["years"]:
            if str(y) not in q:
                bad.append(f"year {y} missing")
    elif t == "range":
        if str(s.period["from"]) not in q or str(s.period["to"]) not in q:
            bad.append("range missing")
    elif t in ("none", "recent"):
        allowed = " ".join([*(lab for _, lab in s.labels), s.item or "", s.measure or ""])
        if any(m.group(0) not in allowed for m in YEAR_RE.finditer(q)):
            bad.append("year written but spec has none")
    for r in s.regions:
        if r not in q and SIDO_SHORT.get(r, r) not in q:
            bad.append(f"region {r} missing")
    for sg in s.sigungu:
        if sg[:-1] not in q:
            bad.append(f"시군구 {sg} missing")
    if not q.strip() or len(q) > 120:
        bad.append("empty or too long")
    if compact(p.raw_name) and compact(p.raw_name) in compact(q):
        bad.append("copies table name")
    return bad


def same_survey_lexical(p: Profile, profiles: dict[str, Profile], banned: set[str], skip: set[str], n: int = 3) -> list[str]:
    """synth_v0's hard negatives: same survey, most shared name words (kept as one source among three)."""
    words = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", p.name))
    scored = []
    for q in profiles.values():
        if q.key in skip or q.key in banned or q.survey != p.survey:
            continue
        ov = len(words & set(re.findall(r"[가-힣A-Za-z0-9]{2,}", q.name)))
        if ov:
            scored.append((-ov, q.key))
    return [k for _, k in sorted(scored)[:n]]


def cmd_build(args) -> int:
    rng = random.Random(args.seed + 2)
    profiles = load_profiles(args.tables)
    src = eval_tables(args.golden)
    banned = set().union(*src.values())
    index = world_index(profiles, banned, args.strict_eval)
    specs = {d["id"]: Spec(**d) for d in _jsonl(OUT / "specs.jsonl")}
    sentences: dict[str, tuple[str, str]] = {}
    for name in ["rule.jsonl", "hcx.jsonl", *sorted(x.name for x in OUT.glob("hcx_*.jsonl")),
                 *sorted(x.name for x in OUT.glob("claude_*.jsonl"))]:
        path = OUT / name
        if not path.exists():
            print(f"  (missing {name})")
            continue
        gen = name.split("_")[0].split(".")[0]
        for r in _jsonl(path):
            if r.get("question"):
                sentences[r["id"]] = (r["question"].strip(), gen)
    rows, dropped = [], Counter()
    seen_q: set[str] = set()
    for sid, s in specs.items():
        if sid not in sentences:
            dropped["no sentence"] += 1
            continue
        q, gen = sentences[sid]
        p = profiles.get(s.anchor)
        if p is None or s.anchor in banned:
            dropped["anchor missing or evaluation table"] += 1
            continue
        if compact(q) in seen_q:
            dropped["duplicate question"] += 1
            continue
        bad = surface_check(q, s, p)
        if bad:
            dropped[bad[0].split(" ")[0] + " " + " ".join(bad[0].split(" ")[-1:])] += 1
            continue
        sets = gold_sets(s, profiles, index)  # recomputed on this machine's tables.jsonl
        pos = [k for k in sets["pos"] if k not in banned]
        if s.anchor not in pos or len(pos) != len(sets["pos"]):
            dropped["gold set touches an evaluation table"] += 1
            continue
        no_neg = [k for k in sets["soft"] if k not in banned]
        neg = [(k, why) for k, why in sets["neg"] if k not in banned]
        neg.sort(key=lambda x: (len(x[1]), rng.random()))  # one slot off first: the closest siblings
        slot_neg = [k for k, _ in neg[:10]]
        lex = same_survey_lexical(p, profiles, banned, set(pos) | set(no_neg) | set(slot_neg))
        seen_q.add(compact(q))
        rows.append({
            "id": sid, "question": q, "anchor": s.anchor, "positives": pos, "no_neg": no_neg,
            "hard_negatives": slot_neg + [k for k in lex if k not in slot_neg],
            "rerank_ok": len(pos) <= RERANK_MAX_POS,
            "slots": {"measure": s.measure, "concept": s.concept, "dims": s.dims, "labels": s.labels,
                      "region_level": s.region_level, "regions": s.regions, "sigungu": s.sigungu,
                      "period": s.period, "cycle": s.cycle, "item": s.item},
            "intent": s.intent, "specificity": s.specificity, "generator": gen, "style": s.style, "version": VERSION,
        })
    # final guard: no evaluation table anywhere
    leak = sum(1 for r in rows for k in [r["anchor"], *r["positives"], *r["no_neg"], *r["hard_negatives"]] if k in banned)
    assert leak == 0, f"{leak} evaluation-table references left"
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    with FINAL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # the training scripts read this list, so they exclude the same tables without importing kosis_rag
    FINAL.with_name("synth_v1_eval_tables.json").write_text(
        json.dumps({"sources": {k: len(v) for k, v in src.items()}, "tables": sorted(banned)}, ensure_ascii=False),
        encoding="utf-8")
    report(rows, dropped, src, banned, profiles, FINAL.with_name("synth_v1_report.md"), rng)
    print(f"{len(rows)} rows -> {FINAL}; dropped {sum(dropped.values())}: {dict(dropped)}")
    return 0


def report(rows, dropped, src, banned, profiles, path: Path, rng: random.Random) -> None:
    def table(counter: Counter, title: str) -> list[str]:
        tot = sum(counter.values()) or 1
        out = [f"\n### {title}\n", "| 값 | 수 | 비율 |", "| --- | --- | --- |"]
        out += [f"| {k} | {v} | {v / tot:.1%} |" for k, v in counter.most_common()]
        return out

    pos_sizes = Counter(min(len(r["positives"]), 21) for r in rows)
    lines = [f"# synth_v1 점검 리포트", "", f"- 행 {len(rows)}개, 표(anchor) {len({r['anchor'] for r in rows})}개",
             f"- 평가 표 제외: 합집합 {len(banned)}표 ("
             + ", ".join(f"{k} {len(v)}" for k, v in src.items()) + "), 데이터 안 평가 표 참조 0 (검사 통과)",
             f"- 리랭커용(정답 {RERANK_MAX_POS}표 이하) {sum(r['rerank_ok'] for r in rows)}행",
             f"- 하드네거티브 평균 {sum(len(r['hard_negatives']) for r in rows) / max(1, len(rows)):.1f}개, "
             f"없음 {sum(1 for r in rows if not r['hard_negatives'])}행",
             f"- 질문에 연도 있음 {sum(bool(YEAR_RE.search(r['question'])) for r in rows) / max(1, len(rows)):.1%}"]
    lines += table(Counter(r["generator"] for r in rows), "생성기")
    lines += table(Counter(r["specificity"] for r in rows), "구체성")
    lines += table(Counter(r["intent"] for r in rows), "의도")
    lines += table(Counter(r["slots"]["period"].get("type") for r in rows), "기간 슬롯")
    lines += table(Counter(profiles[r["anchor"]].survey for r in rows), "조사(anchor)")
    lines += table(Counter("21+" if k == 21 else str(k) for k in pos_sizes.elements()), "정답 집합 크기")
    lines += table(dropped, "버린 행 사유")
    lines += ["\n## 사람 검토용 30문항 (무작위)\n", "| id | 질문 | anchor 표 | 정답 수 | 의도 | 구체성 | 생성기 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rng.sample(rows, min(30, len(rows))):
        lines.append(f"| {r['id']} | {r['question']} | {profiles[r['anchor']].raw_name[:40]} | {len(r['positives'])} | "
                     f"{r['intent']} | {r['specificity']} | {r['generator']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- self test

def selftest() -> int:
    def prof(key, name, mk, dims, regions=None, years=(2020, 2021, 2022), cycle=("Y",), labels=None, level=()):
        return Profile(key=key, name=name, raw_name=name, survey="s", measure=mk[0] if mk else None, mkeys=list(mk),
                       dims=set(dims) | set(level), labels=labels or {}, region_level=set(level), regions=regions,
                       sigungu=[], years=list(years), cycle=set(cycle), months=[], quarters=[], items=[])

    P = {
        "a": prof("a", "시도별 출생아수", ["출생아수"], ["시도"], ["서울특별시", "부산광역시"], level=("시도",)),
        "b": prof("b", "시군구별 출생아수", ["출생아수"], ["시군구"], ["서울특별시"], level=("시도", "시군구")),
        "c": prof("c", "전국 출생아수", ["출생아수"], [], None),
        "d": prof("d", "성별 출생아수", ["출생아수"], ["성"], None, labels={"성": ["남자", "여자"]}),
        "e": prof("e", "시도별 혼인건수", ["혼인건수"], ["시도"], ["서울특별시"], level=("시도",)),
        "f": prof("f", "출생아수 월별", ["출생아수"], [], None, years=(2024, 2025), cycle=("M",)),
    }
    idx = build_index(P)
    s = Spec(id="t", anchor="a", specificity="specific", intent="value", measure="출생아수", mkeys=["출생아수"],
             regions=["서울특별시"], period={"type": "year", "years": [2021]})
    g = gold_sets(s, P, idx)
    assert set(g["pos"]) == {"a", "b"}, g  # both hold 서울 and 2021
    assert "c" in {k for k, _ in g["neg"]} and "e" not in g["pos"] + g["soft"], g
    s2 = Spec(id="t2", anchor="a", specificity="specific", intent="rank", measure="출생아수", mkeys=["출생아수"],
              region_level="시군구", period={"type": "none"})
    assert set(gold_sets(s2, P, idx)["pos"]) == {"b"}
    s3 = Spec(id="t3", anchor="d", specificity="specific", intent="value", measure="출생아수", mkeys=["출생아수"],
              labels=[["성", "여아"]], period={"type": "none"})
    assert "d" in gold_sets(s3, P, idx)["soft"]  # breakdown held, label spelled differently: not a negative
    s4 = Spec(id="t4", anchor="f", specificity="specific", intent="trend", measure="출생아수", mkeys=["출생아수"],
              period={"type": "recent"}, cycle="M")
    assert gold_sets(s4, P, idx)["pos"] == ["f"]
    s5 = Spec(id="t5", anchor="a", specificity="concept", intent="trend", measure="출생아수", mkeys=CONCEPTS["저출산"],
              concept="저출산", period={"type": "none"})
    assert set(gold_sets(s5, P, idx)["pos"]) == {"a", "b", "c", "d", "f"}
    q = "2021년 서울 출생아 수"
    assert surface_check(q, s, P["a"]) == [], surface_check(q, s, P["a"])
    assert surface_check("서울 출생아 수", s, P["a"]) == ["year 2021 missing"]
    rng = random.Random(0)
    assert "2021" in rule_question(rng, s, P["a"])
    print("selftest ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", nargs="?", choices=("plan", "rule", "build"))
    ap.add_argument("--tables", type=Path, default=TABLES)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--strict-eval", action="store_true",
                    help="keep evaluation tables in the index and drop specs whose gold set holds one")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.stage:
        ap.error("stage required (plan / rule / build) or --selftest")
    return {"plan": cmd_plan, "rule": cmd_rule, "build": cmd_build}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
