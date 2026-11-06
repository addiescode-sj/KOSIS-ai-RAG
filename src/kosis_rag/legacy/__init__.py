"""The old retriever-shaped path (T2-41 S2, P2-D18): LangChain BaseRetriever wrappers over the stage functions.

retrievers (BM25TableRetriever, ChromaTableRetriever, HybridRetriever, hybrid_search, ConditionRetriever,
HqRetriever). They keep their last_* attributes for their callers (build_search, backend.app.respond.legacy.answer,
the eval runners, lab); the stage functions they call return that state instead. rerankers (S3: CrossEncoderReranker,
LLMReranker over rerank.stage1 / rerank.stage2), pipeline (S4: build_search(resources)), search and tools (S5: from
chain.search and tools.py; the answer tool is the app's, backend.app.respond.legacy).
"""
