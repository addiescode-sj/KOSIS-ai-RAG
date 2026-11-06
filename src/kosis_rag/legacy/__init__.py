"""The old retriever-shaped path (T2-41 S2, P2-D18): LangChain BaseRetriever wrappers over the stage functions.

retrievers (BM25TableRetriever, ChromaTableRetriever, HybridRetriever, hybrid_search, ConditionRetriever,
HqRetriever). They keep their last_* attributes for their callers (build_search, chain.answer, the eval runners,
lab); the stage functions they call return that state instead. S3 adds rerankers, S5 pipeline / tools / search.
"""
