"""Plan 43 step 2 (Colab): fine-tune the stage-1 re-ranker BAAI/bge-reranker-v2-m3 on the synthetic questions.

Standalone (torch + transformers + numpy only; no other lab file). Reads what lab/rerank_ft_data.py wrote on the PC:
    data/lab/rrft/train_unseen.jsonl, data/lab/rrft/passages.json
Each step, every question gets a group of 1 + NEG tables: its table, then up to TOP_K negatives drawn from the
current-best top 50 (higher ranks more likely, weight 1/sqrt(rank)), up to SIB_K same-name siblings / hard negatives,
filled from the rest of both pools and finally random tables. Loss = softmax cross-entropy over the group
(the gold is index 0). Only the top --layers encoder layers and the classifier head train (fp32 weights, fp16
autocast). --val questions are held out from training (fixed groups, seed 0) and scored before training and after
every epoch: accuracy = gold scores highest in its group, plus mean gold rank in the group.
Each epoch saves only the trained weights:
    data/lab/rrft/<run>/unseen-e<n>.pt  (+ .json with the settings and the val numbers)
The two-stage evaluation later loads the base model and puts these weights back.

    python lab/rerank_ft_train.py --limit 200 --epochs 1     # memory / speed check first
    python lab/rerank_ft_train.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RRFT = ROOT / "data" / "lab" / "rrft"
BASE = "BAAI/bge-reranker-v2-m3"
NEG, TOP_K, SIB_K = 7, 5, 2


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def draw(pool: list[str], k: int, rng: random.Random, weighted: bool, taken: set[str]) -> list[str]:
    """Up to k keys from pool without replacement, skipping taken; weighted -> weight 1/sqrt(position+1)."""
    cand = [(j, x) for j, x in enumerate(pool) if x not in taken]
    out = []
    while cand and len(out) < k:
        w = [1 / math.sqrt(j + 1) if weighted else 1.0 for j, _ in cand]
        pick = rng.choices(range(len(cand)), weights=w)[0]
        out.append(cand.pop(pick)[1])
    return out


def group(row: dict, all_keys: list[str], rng: random.Random) -> list[str]:
    """[gold, NEG negatives]."""
    taken = set(row["ok"])
    negs = draw(row["top"], TOP_K, rng, True, taken)
    taken |= set(negs)
    s = draw(row["sib"], SIB_K, rng, False, taken)
    negs += s
    taken |= set(s)
    for pool in (row["top"], row["sib"]):  # fill when a pool was short
        more = draw(pool, NEG - len(negs), rng, False, taken)
        negs += more
        taken |= set(more)
    while len(negs) < NEG:
        x = rng.choice(all_keys)
        if x not in taken:
            negs.append(x)
            taken.add(x)
    return [row["table"], *negs]


def selftest() -> None:
    rng = random.Random(0)
    keys = [f"T{i}" for i in range(100)]
    row = {"table": "G", "ok": ["G", "A1"], "top": ["A1", "T1", "T2", "T3"], "sib": ["S1"]}
    for _ in range(200):
        g = group(row, keys, rng)
        assert g[0] == "G" and len(g) == 1 + NEG and len(set(g)) == len(g), g
        assert "A1" not in g, "an alt table is a right answer, never a negative"
        assert {"T1", "T2", "T3", "S1"} <= set(g), "short pools are used up before random tables"
    big = {"table": "G", "ok": ["G"], "top": [f"T{i}" for i in range(50)], "sib": ["S1", "S2", "S3"]}
    first = sum(group(big, keys, rng)[1:].count("T0") for _ in range(2000)) / 2000
    last = sum(group(big, keys, rng)[1:].count("T49") for _ in range(2000)) / 2000
    assert first > last, (first, last)
    sib_n = [sum(x.startswith("S") for x in group(big, keys, rng)) for _ in range(200)]
    assert all(n == SIB_K for n in sib_n), "exactly SIB_K siblings when there are enough"
    print(f"selftest ok (top rank 1 drawn {first:.0%}, rank 50 {last:.0%})")


def trainable(model, layers: int) -> list:
    for p in model.parameters():
        p.requires_grad = False
    params = []
    mods = list(model.roberta.encoder.layer[-layers:]) + [model.classifier]
    for m in mods:
        for p in m.parameters():
            p.requires_grad = True
            params.append(p)
    return params


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train_unseen.jsonl", help="file under data/lab/rrft")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--questions", type=int, default=4, help="questions per forward (x 8 pairs)")
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation (questions x accum per update)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--val", type=int, default=200, help="questions held out for the per-epoch check")
    p.add_argument("--limit", type=int, default=0, help="only the first N training questions (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", default=BASE, help="base cross-encoder (name or folder)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = read_jsonl(RRFT / args.data)
    passages = json.loads((RRFT / "passages.json").read_text(encoding="utf-8"))
    all_keys = sorted(passages)
    missing = {k for r in rows for k in [r["table"], *r["top"], *r["sib"]] if k not in passages}
    assert not missing, f"{len(missing)} tables not in passages.json, e.g. {sorted(missing)[:3]}"

    # hold out whole tables for the check, so val questions do not share a table with training
    rng0 = random.Random(0)
    tables = sorted({r["table"] for r in rows})
    rng0.shuffle(tables)
    val_tables, n = set(), 0
    for t in tables:
        if n >= args.val:
            break
        val_tables.add(t)
        n += sum(r["table"] == t for r in rows)
    val = [r for r in rows if r["table"] in val_tables][: args.val]
    train = [r for r in rows if r["table"] not in val_tables]
    if args.limit:
        train = train[: args.limit]
    val_groups = [group(r, all_keys, rng0) for r in val]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).to(device)  # fp32 weights, fp16 autocast
    params = trainable(model, args.layers)
    print(f"device {device}; train {len(train)} questions / val {len(val)} ({len(val_tables)} tables); "
          f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M params")

    def scores(qs: list[str], groups: list[list[str]]):
        a = [q for q, g in zip(qs, groups) for _ in g]
        b = [passages[k] for g in groups for k in g]
        enc = tok(a, b, truncation="only_second", max_length=args.max_length, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**enc).logits.view(len(groups), -1)
        return out.float()

    def check() -> dict:
        model.eval()
        ranks = []
        with torch.no_grad():
            for b in range(0, len(val), 8):
                s = scores([r["q"] for r in val[b:b + 8]], val_groups[b:b + 8]).cpu().numpy()
                ranks += [int((row > row[0]).sum()) + 1 for row in s]
        model.train()
        return {"val_acc": float(np.mean([r == 1 for r in ranks])) if ranks else 0.0,
                "val_mean_rank": float(np.mean(ranks)) if ranks else 0.0}

    run_dir = RRFT / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    before = check()
    print(f"before training: {before}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    per_update = args.questions * args.accum
    total = math.ceil(len(train) / per_update) * args.epochs
    warm = max(1, int(total * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp = torch.amp if hasattr(torch.amp, "GradScaler") else None
    scaler = (amp.GradScaler("cuda", enabled=device.type == "cuda") if amp
              else torch.cuda.amp.GradScaler(enabled=device.type == "cuda"))
    rng = random.Random(args.seed)
    model.train()
    step, started = 0, time.perf_counter()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        opt.zero_grad(set_to_none=True)
        chunks = [order[b:b + args.questions] for b in range(0, len(order), args.questions)]
        for c, chunk in enumerate(chunks, 1):
            bx = [train[i] for i in chunk]
            s = scores([r["q"] for r in bx], [group(r, all_keys, rng) for r in bx])
            loss = torch.nn.functional.cross_entropy(s, torch.zeros(len(bx), dtype=torch.long, device=device))
            scaler.scale(loss / args.accum).backward()
            losses.append(loss.item())
            if c % args.accum == 0 or c == len(chunks):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"  epoch {ep} step {step}/{total} loss {np.mean(losses[-50 * args.accum:]):.4f} "
                          f"({time.perf_counter() - started:.0f}s)")
        after = check()
        ckpt = run_dir / f"unseen-e{ep}.pt"
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, ckpt)
        meta = {"epoch": ep, "loss_mean": float(np.mean(losses)), "steps": step, "seconds": round(time.perf_counter() - started),
                "before": before, "after": after, "base": args.base, "data": args.data, "train_questions": len(train),
                "val_questions": len(val), "layers": args.layers, "lr": args.lr, "questions": args.questions,
                "accum": args.accum, "group": 1 + NEG, "top_k": TOP_K, "sib_k": SIB_K, "max_length": args.max_length,
                "seed": args.seed, "limit": args.limit}
        ckpt.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> {ckpt} (epoch loss {meta['loss_mean']:.4f}; val {after})")
    print(f"-> {run_dir}  (copy the folder to the PC: data/lab/rrft/{run_dir.name}/)")


if __name__ == "__main__":
    main()
