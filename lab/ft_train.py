"""Plan 29: contrastive fine-tuning of bge-m3 on the synthetic questions, with sibling hard negatives (step 1 of 2).

Pairs: data/lab/synth/synth_v0.jsonl question -> its table's V0 passage (n2-L60 embed text). Two regimes:
    unseen  rows whose table is a gold / alt table of routing_v0 or retrieval_golden are dropped (new-table test)
    seen    every row (the deployed setting: the table list is fixed)
Hard negatives per row: the file's hard_negatives + up to MINED tables V0 dense ranks above the gold (top 10),
minus alt_tables, minus same-name siblings the question cannot tell apart (no year in the question, or a year
inside the sibling's period). One is drawn per epoch; other rows' positives in the batch are negatives too.
Loss InfoNCE (temperature TAU). Only the top --layers encoder layers train (embeddings and lower layers frozen:
full fine-tuning does not fit 8GB with Adam). Each epoch saves only the trained weights:
data/lab/ft/<run>/<regime>-e<n>.pt (+ .json with the settings). ft_eval.py reads them.

    python lab/ft_train.py
    python lab/ft_train.py --regime unseen --batch 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from lab_common import (
    DEFAULT_MODEL, RETRIEVAL, ROOT, ROUTING, SYNTH, doc_vectors, docs, load_records, query_text, question_vectors,
    read_jsonl, source_sha,
)
from normalize_text import build_doc

FT_DIR = ROOT / "data" / "lab" / "ft"
TAU = 0.02
MINED = 3
YEAR = re.compile(r"(?<!\d)(19\d\d|20\d\d)(?!\d)")


def eval_tables() -> set[str]:
    """Every table a golden row accepts: routing gold + alt, retrieval required groups."""
    out = set()
    for it in read_jsonl(ROUTING):
        out |= {Path(it["table"]).stem, *it.get("alt_tables", [])}
    for it in read_jsonl(RETRIEVAL):
        out |= {k for g in it.get("required") or [] for k in g}
    return out


def table_info(records: list[dict]) -> dict[str, dict]:
    return {r["info"]["key"]: {"name": build_doc(r)["fields"]["name"], "start": r.get("start"), "end": r.get("end")}
            for r in records}


def indistinguishable(question: str, gold: str, neg: str, info: dict[str, dict]) -> bool:
    """A same-name sibling the question cannot tell apart from the gold table (would be a false negative)."""
    if info[neg]["name"] != info[gold]["name"]:
        return False
    years = [int(y) for y in YEAR.findall(question)]
    s, e = info[neg]["start"], info[neg]["end"]
    if not years or s is None:
        return True
    return any(s <= y <= (e if e is not None else s) for y in years)


def build_examples(regime: str, model: str, max_labels: int, max_tokens: int, batch: int, state: dict) -> tuple[list[dict], dict]:
    records, sha = load_records(), source_sha()
    ds = docs(records, "norm", max_labels)
    keys = [d["key"] for d in ds]
    index = {k: i for i, k in enumerate(keys)}
    passage = {d["key"]: d["embed"] for d in ds}
    info = table_info(records)
    synth = read_jsonl(SYNTH)
    held = eval_tables()

    # mining with the V0 model (cached vectors, nothing new to embed)
    dv = doc_vectors(model, "norm", max_labels, max_tokens, ds, sha, state, batch)
    sn = [query_text(r["question"], "norm") for r in synth]
    sv = question_vectors(model, sn, SYNTH.stem, state, batch)
    state.clear()  # free the inference copy before training loads its own

    rows, dropped_fn = [], 0
    for r, q, v in zip(synth, sn, sv):
        if regime == "unseen" and (r["table"] in held or any(a in held for a in r.get("alt_tables", []))):
            continue
        ok = {r["table"], *r.get("alt_tables", [])}
        order = np.argsort(-(dv @ v))[:10]
        above = []
        for j in order:
            k = keys[j]
            if k == r["table"]:
                break
            above.append(k)
        pool = list(dict.fromkeys([*r.get("hard_negatives", []), *above[:MINED]]))
        keep = [k for k in pool if k in index and k not in ok and not indistinguishable(q, r["table"], k, info)]
        dropped_fn += len([k for k in pool if k in index and k not in ok]) - len(keep)
        rows.append({"q": q, "table": r["table"], "ok": sorted(ok), "negs": keep})
    stats = {"regime": regime, "rows": len(rows), "tables": len({x["table"] for x in rows}),
             "rows_without_negative": sum(1 for x in rows if not x["negs"]), "false_negatives_dropped": dropped_fn,
             "held_tables": len(held)}
    return [{**x, "pos_text": passage[x["table"]], "neg_texts": {k: passage[k] for k in x["negs"]}} for x in rows], stats


def trainable(model, layers: int) -> list:
    """Freeze everything, then open the top `layers` encoder layers of the XLM-R backbone."""
    enc = model[0].auto_model
    for p in enc.parameters():
        p.requires_grad = False
    params = []
    for layer in enc.encoder.layer[-layers:]:
        for p in layer.parameters():
            p.requires_grad = True
            params.append(p)
    return params


def trained_state(model) -> dict:
    enc = model[0].auto_model
    return {n: p.detach().cpu() for n, p in enc.named_parameters() if p.requires_grad}


def load_ft(base: str, ckpt: Path, max_tokens: int):
    """Base bge-m3 with the saved top layers put back (inference, fp16 on GPU)."""
    import torch
    from lab_common import load_model

    model = load_model(base, max_tokens)
    sd = torch.load(ckpt, map_location="cpu")
    result = model[0].auto_model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in sd.items()},
                                                  strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    return model


def embed(model, texts: list[str], max_len: int, device):
    """Training forward (with grad). Unit vectors even if the model has no Normalize module."""
    import torch

    model.max_seq_length = max_len
    prep = getattr(model, "preprocess", None) or model.tokenize  # sentence-transformers 6 renamed tokenize
    feats = prep(texts)
    feats = {k: v.to(device) if hasattr(v, "to") else v for k, v in feats.items()}  # newer versions add str entries
    return torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)


def train(regime: str, args, run_dir: Path) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    state: dict = {}
    examples, stats = build_examples(regime, args.model, args.max_labels, args.max_tokens, args.batch, state)
    print(json.dumps(stats, ensure_ascii=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model, device=str(device))  # fp32 weights, fp16 autocast
    params = trainable(model, args.layers)
    model.train()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(examples) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    warm = max(1, int(total * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warmup, then linear decay to 0
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp = torch.amp if hasattr(torch.amp, "GradScaler") else None  # torch < 2.3 only has torch.cuda.amp
    scaler = (amp.GradScaler("cuda", enabled=device.type == "cuda") if amp
              else torch.cuda.amp.GradScaler(enabled=device.type == "cuda"))
    rng = random.Random(args.seed)
    step, started = 0, time.perf_counter()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        run_loss = []
        for b in range(0, len(order), args.batch):
            bx = [examples[i] for i in order[b:b + args.batch]]
            negs = [rng.choice(list(x["neg_texts"].items())) if x["neg_texts"] else None for x in bx]
            doc_keys = [x["table"] for x in bx] + [n[0] for n in negs if n]
            doc_texts = [x["pos_text"] for x in bx] + [n[1] for n in negs if n]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                qe = embed(model, [x["q"] for x in bx], 64, device)
                de = embed(model, doc_texts, args.doc_tokens, device)
                logits = (qe @ de.T).float() / TAU
            # mask documents that are also right for a row (same table / alt), except the row's own positive
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i, x in enumerate(bx):
                ok = set(x["ok"])
                for j, k in enumerate(doc_keys):
                    if j != i and k in ok:
                        mask[i, j] = True
            logits = logits.masked_fill(mask, -1e4)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(bx), device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss.append(loss.item())
            if step % 50 == 0:
                print(f"  {regime} epoch {ep} step {step}/{total} loss {np.mean(run_loss[-50:]):.4f} "
                      f"({time.perf_counter() - started:.0f}s)")
        ckpt = run_dir / f"{regime}-e{ep}.pt"
        torch.save(trained_state(model), ckpt)
        meta = {**stats, "epoch": ep, "loss_mean": float(np.mean(run_loss)), "steps": step,
                "seconds": round(time.perf_counter() - started), "model": args.model, "layers": args.layers,
                "lr": args.lr, "batch": args.batch, "tau": TAU, "doc_tokens": args.doc_tokens, "seed": args.seed,
                "source_sha": source_sha()[:8]}
        ckpt.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> {ckpt} (epoch loss {meta['loss_mean']:.4f})")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--regime", choices=("both", "unseen", "seen"), default="both")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--doc-tokens", type=int, default=512)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_dir = FT_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    for regime in (("unseen", "seen") if args.regime == "both" else (args.regime,)):
        train(regime, args, run_dir)
    print(f"-> {run_dir}  (next: python lab/ft_eval.py --run {run_dir.name})")


if __name__ == "__main__":
    main()
