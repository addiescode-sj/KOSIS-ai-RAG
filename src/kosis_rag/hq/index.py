"""Load a question build into Chroma (T2-39): one Chroma document per hypothetical question.

    set PYTHONPATH=.;src;vendor/phase1;vendor/phase1/src          (Windows cmd)
    python -m kosis_rag.hq.index --build <build_id> --ckpt data/models/unseen-e1.pt
    python -m kosis_rag.hq.index --build <build_id> --ckpt data/models/unseen-e1.pt --desc   # S1-desc ablation

Questions are embedded with the same fine-tuned embedder as the service's table collection after the same surface
rules (text.norm_query). Collection `kosis_hq__<build_id>__<ckpt8>`; its metadata records embed_model, embed_ckpt,
doc_format "hq" and hq_build, and `HqRetriever.connect` refuses a collection embedded with other weights than the
dense side. `--desc` indexes each table's one-sentence description instead (build id `<build_id>-desc`).

T2-41 S2: only this CLI is left here (S6a moves it to kosis_rag_build/hq/questions_index.py). Searching the index is
kosis_rag.retrieve.hq (HqIndex, max_pool, hq_bm25, ckpt8, collection_name; HqRetriever in kosis_rag.legacy.retrievers),
reading a build kosis_rag.adapters.files.read_questions.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime
from pathlib import Path

from kosis_rag.adapters import files
from kosis_rag.retrieve import hq as hq_search  # module access: this CLI re-exports none of its names
from kosis_rag.retrieve.constants import HQ_DOC_FORMAT
from kosis_rag.schemas.resources import ChromaConfig
from kosis_rag.shared import constants
from kosis_rag.shared import tabletext as text

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a hypothetical question build into Chroma (T2-39).")
    parser.add_argument("--build", required=True, help="build id printed by python -m kosis_rag.hq.build")
    parser.add_argument("--ckpt", default=None, help="fine-tuned bge-m3 .pt (default: EMBED_CKPT in .env)")
    parser.add_argument("--desc", action="store_true", help="index the descriptions instead (build id <id>-desc)")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--collection", default=None, help="override the collection name")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    from kosis_rag.adapters import models as model
    from kosis_rag.adapters.chroma import chroma_client
    from kosis_rag.index import load

    load_dotenv(REPO / ".env")
    build = args.build + (constants.DESC_SUFFIX if args.desc and not args.build.endswith(constants.DESC_SUFFIX) else "")
    rows = files.read_questions(build)
    if not rows:
        raise SystemExit(f"build {build} has no rows")
    embedder = model.load_embedder(args.ckpt or os.getenv("EMBED_CKPT"), device=args.device, doc_format=text.DOC_FORMAT)
    texts = [text.norm_query(r["text"]) for r in rows]
    vectors = embedder.embed_documents(texts)
    name = args.collection or hq_search.collection_name(build, embedder.ckpt)
    digest = hashlib.sha256("\n".join(f"{r['id']}\t{r['text']}" for r in rows).encode("utf-8")).hexdigest()
    record = {"index_version": datetime.now().strftime("%Y%m%d%H%M"), "doc_format": HQ_DOC_FORMAT, "hq_build": build,
              "embed_model": embedder.name, "embed_ckpt": embedder.ckpt, "data_sha256": digest,
              "dim": len(vectors[0]), "normalized": True, "hnsw:space": "cosine", "n_docs": len(rows),
              "n_tables": len({r["key"] for r in rows})}
    # CLI: CHROMA_HOST / CHROMA_PORT from the environment or .env (T2-41 S4: inlined from legacy.retrievers, which no
    # longer reads the environment; this CLI is a listed exception until S6a moves it to kosis_rag_build)
    chroma = ChromaConfig(host=os.getenv("CHROMA_HOST") or "localhost", port=int(os.getenv("CHROMA_PORT") or 8001))
    count = load(chroma_client(chroma), [r["id"] for r in rows], texts, [{"key": r["key"], "type": r["type"]} for r in rows],
                 vectors, name, record)
    print(f"collection {name}: {count} questions over {record['n_tables']} tables (embedder {embedder.ckpt})")
    print(f"to search with it, set HQ_BUILD={build} in .env (empty = off); HQ_BM25=on adds the question BM25")
    return 0 if count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
