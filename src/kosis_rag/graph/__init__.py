"""The search as a LangGraph graph (T2-41 S4: from graph.py): search (node table, edges, build / run / stream),
nodes_api (the node API's node set, apply_update, ground_docs) and spans (per-node timing).

LangGraph is imported only in this package (within kosis_rag). Nothing is re-exported here: import from the module
that defines the name, e.g. `kosis_rag.graph.search.build_graph`.
"""
