"""RAG answer: retrieve candidate tables -> prompt -> Phase 1 LLMClient -> verify -> add citations by code.

The model only writes a guide sentence over the candidates. Table ids, periods and citations come from the catalog;
a reply that names an id or year not in the candidates, or a failed model call, falls back to a code template.
"""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from kosis_chat.llm.model import STATUS_OK, LLMClient, LLMResult

from kosis_rag.loader import to_candidate
from kosis_rag.schemas import SearchResult, TableCandidate

NO_MATCH = "NO_MATCH"
REFUSAL_TEXT = "제공된 KOSIS 통계표 중에서 이 질문에 맞는 표를 찾지 못했습니다. 다른 표현으로 다시 질문해 주세요."
# TODO(T2-14, T2-09): refuse on period / frequency mismatch and calibrate a score threshold; today only "no hit" refuses.

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 KOSIS 통계표 안내 도우미입니다. 아래 후보 통계표 목록만 근거로, 질문에 맞는 표를 1~3개 골라 왜 맞는지 짧게 설명하세요.\n"
     "규칙: 목록에 없는 표·기간·숫자를 지어내지 마세요. 통계 수치는 말하지 마세요. 표를 가리킬 때는 [번호]와 표 ID를 쓰세요.\n"
     f"맞는 표가 하나도 없으면 다른 말 없이 {NO_MATCH} 만 출력하세요."),
    ("human", "후보 통계표:\n{context}\n\n질문: {question}"),
])

_TBL_ID = re.compile(r"\bDT_[0-9A-Za-z_]+")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def format_context(candidates: list[TableCandidate]) -> str:
    return "\n".join(
        f"[{i}] {c.tbl_id} | {c.table_name} | {c.survey_name} | {c.organization} | {c.period}"
        for i, c in enumerate(candidates, 1)
    )


def llm_runnable(client: LLMClient) -> Runnable:
    """Wrap a Phase 1 client (HCX / mock) as a LangChain runnable: prompt messages in, LLMResult out."""

    def call(prompt_value) -> LLMResult:
        messages = prompt_value.to_messages()
        system = "\n".join(m.content for m in messages if m.type == "system")
        user = "\n".join(m.content for m in messages if m.type != "system")
        return client.generate(system, user)

    return RunnableLambda(call, name=f"phase1_llm:{client.name}")


def unsupported_claims(text: str, question: str, candidates: list[TableCandidate]) -> list[str]:
    """Table ids and years in the reply that neither the candidates nor the question contain."""
    ranges = [(min(ys), max(ys)) for c in candidates if (ys := [int(y) for y in _YEAR.findall(c.period)])]
    asked = set(_YEAR.findall(question))
    ids = [t for t in _TBL_ID.findall(text) if t not in {c.tbl_id for c in candidates}]
    years = [y for y in _YEAR.findall(text) if y not in asked and not any(lo <= int(y) <= hi for lo, hi in ranges)]
    return ids + years


def template_answer(candidates: list[TableCandidate]) -> str:
    lines = ["질문과 관련 있어 보이는 통계표입니다."]
    lines += [f"[{i}] {c.table_name} ({c.tbl_id}) · {c.organization} · {c.period}" for i, c in enumerate(candidates[:3], 1)]
    return "\n".join(lines)


def with_citations(text: str, candidates: list[TableCandidate]) -> str:
    return text + "\n\n출처:\n" + "\n".join(f"- {c.to_source().citation()}" for c in candidates[:3])


def search(retriever: BaseRetriever, question: str, k: int = 5) -> list[TableCandidate]:
    return [to_candidate(d) for d in retriever.invoke(question)[:k]]


def answer(retriever: BaseRetriever, client: LLMClient, question: str, k: int = 5) -> SearchResult:
    candidates = search(retriever, question, k)
    if not candidates:
        return SearchResult(question=question, no_answer=True, answer=REFUSAL_TEXT, answer_origin="refusal")

    result: LLMResult = (PROMPT | llm_runnable(client)).invoke(
        {"question": question, "context": format_context(candidates)}, config={"run_name": "table_guide_chain"}
    )
    text = result.text.strip()
    if result.status == STATUS_OK and text == NO_MATCH:
        return SearchResult(question=question, candidates=candidates, no_answer=True, answer=REFUSAL_TEXT,
                            answer_origin="refusal", llm_status=result.status)
    origin = "llm"
    if result.status != STATUS_OK or not text or unsupported_claims(text, question, candidates):
        text, origin = template_answer(candidates), "template"
    return SearchResult(question=question, candidates=candidates, answer=with_citations(text, candidates),
                        answer_origin=origin, llm_status=result.status)
