"""Ask the local store a question and print the top-k tables.

    python lab/chroma_query.py "최근 출생아 수 추이"
    python lab/chroma_query.py "시도별 인구" --text raw --k 10
    python lab/chroma_query.py "시도별 인구" --min-end 2023 --cycle M
"""

from __future__ import annotations

import argparse

from chroma_common import DEFAULT_MODEL, TEXTS, client, encode, load_model, open_collection, query_text


def search(col, model, model_name: str, question: str, text: str, k: int, where: dict | None = None) -> list[dict]:
    q = encode(model, model_name, [query_text(question, text)], "query")[0]
    res = col.query(query_embeddings=[q.tolist()], n_results=k, where=where, include=["metadatas", "distances"])
    # never query_texts=: Chroma would embed the question with its own default model, not ours
    # cosine distance = 1 - cosine similarity
    return [{"key": i, "sim": 1 - d, **m} for i, d, m in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--text", choices=TEXTS, default="norm")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min-end", type=int, default=None, help="metadata filter: tables whose last year >= this")
    p.add_argument("--cycle", default=None, help="metadata filter: tables with this cycle (Y, H, Q, M, ...)")
    args = p.parse_args()

    col = open_collection(client(), args.model, args.text, args.max_labels)
    if col is None:
        raise SystemExit(f"no collection yet: python lab/chroma_build.py --text {args.text}")
    conds = [c for c in ({"end": {"$gte": args.min_end}} if args.min_end else None,
                         {"cycle": {"$contains": args.cycle}} if args.cycle else None) if c]
    where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)
    model = load_model(args.model, 512)
    for rank, hit in enumerate(search(col, model, args.model, args.question, args.text, args.k, where), 1):
        print(f"{rank:2d}. {hit['sim']:.3f}  {hit['key']}  {hit['table_name']}  ({hit.get('start')}~{hit.get('end')}, {hit['organization']})")


if __name__ == "__main__":
    main()
