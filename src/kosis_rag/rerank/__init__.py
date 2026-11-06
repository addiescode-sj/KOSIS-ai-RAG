"""Stage 3 of the search (T2-41 S3, from rerank.py, hcx.py and clova.py): the two re-ranking stages after fuse.

    fused pool (retrieve)
      -> stage1: a fine-tuned cross-encoder (bge-reranker-v2-m3) scores (question, table passage) for the top n;
         mode "blend" = RRF of the re-ranker order and the input order, "score" = re-ranker only (rerank_stage1).
      -> stage2: an LLM re-ranker sees the top k_in table keys and returns the ones it picks; picked tables go first
         in their stage-1 order, then the rest. Keys it invents are dropped. A failed call keeps the stage-1 order
         and records the error (rerank_stage2).

stage1 (blend_order, passage_scorer, rerank_stage1), stage2 (rerank_stage2, Score, Select), selectors (the stage 2
factories over an LLMConfig: clova, hcx, gpt -> LLM_RERANKERS), hcx (HcxSelector, parse, user_message, REASONS; its
system prompt is prompts/stage2-hcx-v3.md, package data), constants, schemas (the node In/Out), node (rerank1,
rerank2). The LangChain retriever wrappers are in legacy/rerankers.py. rerank imports schemas, shared, the stages
before it and llm only.
"""
