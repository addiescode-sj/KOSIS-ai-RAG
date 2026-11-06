"""Refusal sentences the code picks, never the LLM (T2-41 S3: from chain.py).

REFUSALS maps why the stage 2 LLM re-ranker picked nothing (rerank.hcx.REASONS) to its sentence; a reason without
its own sentence (no_item, None) gets REFUSAL_TEXT: refusal_text = REFUSALS.get(reason, REFUSAL_TEXT).
Only out_of_scope gets its own sentence: on the T2-09 golden set HCX-007 labelled 9 of 13 out-of-scope refusals
"no_item", so a "related tables exist" sentence would mislead; no_item falls back to REFUSAL_TEXT.
The module reads them in api.to_output (SearchOutput.refusal_text). T2-41 S6b: NO_MATCH (the app's guide LLM
sentinel) moved to backend/app/respond/texts.py: the module never used it.
"""

from __future__ import annotations

REFUSAL_TEXT = "제공된 KOSIS 통계표 중에서 이 질문에 맞는 표를 찾지 못했습니다. 다른 표현으로 다시 질문해 주세요."
REFUSALS = {
    "out_of_scope": "이 질문은 제공된 KOSIS 인구 통계표가 다루는 범위 밖입니다. 다른 통계(예: 지역·경제·보건 통계)는 KOSIS에서 직접 찾아 주세요.",
}
# ponytail: no score threshold in the service yet (tau is calibrated only in evals); refusals are "no candidates" and
# the stage 2 LLM's empty pick (T2-09). A filter that removes every table refuses with the generic text.
