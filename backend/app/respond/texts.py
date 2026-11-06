"""Fixed sentences of the guide answer, written by code (T2-41 S5: from chain.template_answer / with_citations).

The refusal sentences are the module's (kosis_rag.gate.texts, read through SearchOutput.refusal_text).
NO_MATCH is what the guide LLM answers when no candidate fits (prompts/guide-v1.md asks for it; node.py and
verify.py read it); T2-41 S6b moved it here from kosis_rag.gate.texts, which never used it.
"""

from __future__ import annotations

NO_MATCH = "NO_MATCH"  # the guide LLM's "no table fits" sentinel (guide-v1.md)
TEMPLATE_OPENING = "질문과 관련 있어 보이는 통계표입니다."  # first line of template_answer
CITATION_HEADER = "출처:"  # the line before the source list (with_citations)
