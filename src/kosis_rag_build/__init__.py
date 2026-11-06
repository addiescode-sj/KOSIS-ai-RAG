"""Offline DB preparation for the search module (T2-41 S6a, P2-D3 / P2-D21): what is built before kosis_rag searches.

    tables.py      data/tables.jsonl (the table registry) from the provided data
    sigungu.py     data/sigungu.json (시군구 -> 시도) from one 읍면동 table
    chroma.py      the table documents and metadata, embedded and loaded into Chroma (or written to a .npz)
    hq/            T2-39 hypothetical questions per table (generation, rules, Doc2Query−− filter, report) and the
                   question index load (hq/questions_index.py)
    graph/         the LangGraph graphs over those nodes (index_build.py: tables / sigungu / chroma / questions;
                   hq_build.py: the question generation)
    cli.py         python -m kosis_rag_build.cli {tables|sigungu|chroma|text|hq-build|hq-index} (arguments, then
                   the shell environment; no .env)

This package may import kosis_rag.{schemas,shared,adapters,llm,rerank.stage1}; kosis_rag never imports it
(tests/test_layout.py). The search reads what this package writes (adapters.files, retrieve.hq).
"""
