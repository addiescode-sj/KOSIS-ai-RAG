"""Reverse HyDE (T2-39): hypothetical questions per table, filtered Doc2Query−− style, as a separate search index.

    offline   python -m kosis_rag.hq.build   tables -> generation graph (graph.py) -> data/hq/<build_id>/questions.jsonl
              python -m kosis_rag.hq.index   questions -> Chroma collection kosis_hq__<build_id>__<ckpt8>
    query     HQ_BUILD=<build_id>            the search graph adds retrieve_hq (and retrieve_hq_bm25 with HQ_BM25=on)

The table documents (n2) are not changed: questions live in their own index, and a table found through its questions
is handed on with its n2 text, so the re-rankers see what they were trained on (plans/T2-39.md).
"""
