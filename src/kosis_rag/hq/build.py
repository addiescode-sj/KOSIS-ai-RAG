"""Build the hypothetical question set (T2-39): python -m kosis_rag.hq.build.

    set PYTHONPATH=.;src;vendor/phase1;vendor/phase1/src          (Windows cmd; bash: PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src)
    python -m kosis_rag.hq.build --model gpt-4o-mini --limit 20          # a sample: read report.md before the full run
    python -m kosis_rag.hq.build --model gpt-4o-mini --concurrency 8     # every table (cached replies are reused)
    python -m kosis_rag.hq.build --no-score                              # rules only (build id ...-f0)

OPENAI_API_KEY comes from the repo's .env and is never printed. The stage 1 cross-encoder (--rerank-ckpt, default
RERANK_CKPT in .env) scores each question against its own table for the Doc2Query−− filter. Prices come from
evals/pricing.json (gpt-4o-mini; null = cost not shown). Output: data/hq/<build_id>/ (questions.jsonl, report.md, ...).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kosis_rag.hq import graph as hq_graph
from kosis_rag.hq import prompt as hq_prompt
from kosis_rag.shared import tabletext as text

REPO = Path(__file__).resolve().parents[3]


def pair_scorer(ckpt: str, records: list, fmt: str, device: str | None = None) -> hq_graph.PairScore:
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
    print(f"stage 1 scorer: {scorer.ckpt}")
    return lambda key, questions: [score(q, [key])[0] for q in questions]


def cost_line(model: str, totals: dict) -> str:
    """Tokens and cost from evals/pricing.json ('not set' when the price is null).

    Side effects: reads evals/pricing.json. Node: none (CLI output).
    """
    try:
        from evals.ops import cost, load_pricing
    except ImportError:  # evals is not on PYTHONPATH
        return "cost: evals.ops not importable (add . to PYTHONPATH)"
    pricing = load_pricing()
    unit = (pricing.get(model) or {}).get("currency", "")
    parts = []
    for label, i, o in (("this run", totals["new_input_tokens"], totals["new_output_tokens"]),
                        ("all replies", totals["input_tokens"], totals["output_tokens"])):
        c = cost(model, i, o, pricing)
        parts.append(f"{label}: input {i:,} / output {o:,} tokens, cost "
                     + (f"{c:.4f} {unit}" if c is not None else f"not set (fill {model} in evals/pricing.json)"))
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate, check and filter hypothetical questions per table (T2-39).")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None, help="first N tables of tables.jsonl only (a sample build)")
    parser.add_argument("--concurrency", type=int, default=hq_graph.CONCURRENCY)
    parser.add_argument("--tables", type=Path, default=REPO / "data" / "tables.jsonl")
    parser.add_argument("--rerank-ckpt", default=None, help="stage 1 cross-encoder (default: RERANK_CKPT in .env)")
    parser.add_argument("--no-score", action="store_true", help="rules only, no score filter (build id ...-f0)")
    parser.add_argument("--frac", type=float, default=hq_graph.FILTER_FRAC, help="global share dropped by score")
    parser.add_argument("--min-keep", type=int, default=hq_graph.MIN_KEEP, help="questions kept per table at least")
    parser.add_argument("--text-format", default=text.DOC_FORMAT, help="table text format (n2-L60)")
    parser.add_argument("--device", default=None, help="cpu or cuda for the cross-encoder (default: auto)")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    from kosis_rag.adapters.files import read_records
    from kosis_rag.llm.config import LLMConfig
    from kosis_rag.llm.openai import OpenAIClient

    load_dotenv(REPO / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("warning: OPENAI_API_KEY is empty: only tables already in the cache will succeed")
    records = read_records(args.tables)
    sha = hq_graph.tables_sha(args.tables)
    scorer = None
    if not args.no_score:
        ckpt = args.rerank_ckpt or os.getenv("RERANK_CKPT")
        if not ckpt:
            raise SystemExit("set --rerank-ckpt (or RERANK_CKPT in .env), or pass --no-score")
        scorer = pair_scorer(ckpt, records, args.text_format, args.device)
    client = OpenAIClient(LLMConfig(provider="openai", api_key=os.getenv("OPENAI_API_KEY") or None, model=args.model,
                                    temperature=hq_prompt.TEMPERATURE, max_tokens=hq_prompt.MAX_TOKENS,
                                    seed=hq_prompt.SEED))
    n = min(args.limit or len(records), len(records))
    print(f"prompt {hq_prompt.PROMPT_VERSION} ({hq_prompt.prompt_hash()[:8]}), model {args.model}, "
          f"{n} tables, concurrency {args.concurrency}, cache {hq_graph.CACHE_DIR / hq_prompt.prompt_hash()[:8]}")
    state = hq_graph.run_build(records, client, model=args.model, tables_sha256=sha, limit=args.limit,
                               frac=args.frac, min_keep=args.min_keep, scorer=scorer, concurrency=args.concurrency,
                               fmt=args.text_format)
    t = state.totals
    print(f"build {state.build_id} -> {state.out_dir}")
    print(f"tables {t['tables']} (failed {t['failed']}), questions generated {t['generated']}, "
          f"rule-dropped {t['rule_dropped']}, score-dropped {t['score_dropped']}, kept {t['kept']}")
    print(f"replies {t['replies']} (API calls this run {t['calls']}, cached {t['cached']}), API {t['api_seconds']}s")
    print(cost_line(args.model, t))
    print(f"next: python -m kosis_rag.hq.index --build {state.build_id} --ckpt <EMBED_CKPT>, "
          f"then HQ_BUILD={state.build_id} in .env")
    return 0 if t["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
