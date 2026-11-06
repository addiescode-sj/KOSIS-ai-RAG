"""KOS-88 analysis 3: real bge-m3 token counts of the texts we would embed (characters are only a proxy).

    python lab/token_lengths.py
    python lab/token_lengths.py --max-labels 100
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from chroma_common import DEFAULT_MODEL, TEXTS, docs, load_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    records = load_records()
    print(f"{len(records)} tables, tokenizer {args.model}")
    print("text   median   p90   p99    max  >512  >1024  >8192  tokens/char")
    for text in TEXTS:
        texts = [d["embed"] for d in docs(records, text, args.max_labels)]
        n = sorted(len(ids) for ids in tok(texts, add_special_tokens=True)["input_ids"])
        q = lambda f: n[min(len(n) - 1, int(len(n) * f))]  # noqa: E731
        ratio = sum(n) / sum(map(len, texts))
        print(f"{text:5s} {q(.5):7d} {q(.9):5d} {q(.99):5d} {n[-1]:6d} {sum(x > 512 for x in n):5d} {sum(x > 1024 for x in n):6d}"
              f" {sum(x > 8192 for x in n):6d}  {ratio:.2f}")


if __name__ == "__main__":
    main()
