"""Reverse HyDE (T2-39): hypothetical questions per table, filtered Doc2Query−− style, as a separate search index.

    offline   python -m kosis_rag_build.cli hq-build   tables -> generation graph (graph/hq_build.py over hq/nodes.py)
                                                       -> data/hq/<build_id>/questions.jsonl
              python -m kosis_rag_build.cli hq-index   questions -> Chroma collection kosis_hq__<build_id>__<ckpt8>
    query     IndexSpec(hq_build=<build_id>)           the search graph adds retrieve_hq (and retrieve_hq_bm25 with
                                                       SearchOptions(hq_bm25=True)); kosis_rag.retrieve.hq reads it

The table documents (n2) are not changed: questions live in their own index, and a table found through its questions
is handed on with its n2 text, so the re-rankers see what they were trained on (plans/T2-39.md).
T2-41 S6a: from the search package's old hq/ (prompt.py, rules.py, schemas.py, graph.py, build.py, index.py).
"""
