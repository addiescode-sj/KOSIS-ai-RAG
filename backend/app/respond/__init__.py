"""The app's answer layer (T2-41 S5: from src/kosis_rag/chain.py): what the backend says after the search module's gate.

The module (kosis_rag) ends at gate with its route and SearchOutput (graph.gate_to_api names it for the API). This package writes the sentence:
refuse (code text from SearchOutput.refusal_text, no LLM) or guide (the guide LLM over the candidates, checked by
code, citations added by code, streamed piece by piece). graph assembles the module's nodes and these into the API's
graph (/search, /search/stream, /nodes/*), whose state is state.SearchState (route refuse | guide, result).

    texts     fixed sentences (template opening, citation header)
    patterns  the regexes the check reads (table ids, years, [n] references)
    prompts   the guide prompt (prompts/guide-v1.md, read through importlib.resources)
    verify    unsupported_claims, references, picked_from_reply, streamable_prefix
    render    format_context, template_answer, with_citations
    node      refuse / guide (step functions), GuideClient, answer_nodes (graph node adapters)
    state     the API's SearchState
    graph     gate_to_api, build_answer_graph, api_nodes, run_answer, stream_answer
    legacy    the old retriever-shaped answer (answer, the answer_table_question tool)
"""
