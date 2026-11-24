"""Query rewriting (KOS-192): a question in everyday words -> one search question in official statistics terms.

    rerank1 (stage 1 top score) -> [rewrite: one LLM call] -> check (code, no LLM) -> searched again with the rewrite

constants (prompt file, token and length limits), schemas (RewriteInput, RewriteOutput), node (rewrite_question: the
LLM call and its reply; check_rewrite: the rules that throw a rewrite away; user_message, parse_reply,
rewrite_client). The system prompt is prompts/rewrite-v1.md (package data, read through importlib.resources).
Steps 1-2 of KOS-192 use it from offline scripts only; the search graph does not call it yet. rewrite imports schemas,
shared, the stages before it (parse: ParsedQuery) and llm only (tests/test_layout.py).
"""
