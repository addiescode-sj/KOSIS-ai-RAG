"""Stage 2 of the search (T2-41 S2, from retriever.py and hq/index.py): the ranked lists and their fusion.

dense (DenseIndex over a Chroma table collection), bm25 (Bm25Index over the table text), hq (HqIndex over a question
collection, max_pool, hq_bm25), lists (condition_filter, search_list), fuse (rrf_fuse), constants, schemas (the node
In/Out), node (embed, retrieve_list, fuse). No LangChain retriever class here (legacy/retrievers.py wraps these) and
no self state: every search returns what it removed / relaxed. retrieve imports schemas, shared and parse only.
"""
