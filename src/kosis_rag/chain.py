"""RAG answer: retrieve candidate tables -> prompt -> LLM (ChatClovaX or Phase 1 LLMClient) -> verify -> add citations by code.

The model only writes a guide sentence over the candidates. Table ids, periods and citations come from the catalog;
a reply that names an id or year not in the candidates, or a failed model call, falls back to a code template.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from kosis_chat.config import get_secret, load_env, load_settings
from kosis_chat.llm.model import STATUS_API_ERROR, STATUS_OK, LLMClient, LLMResult

from kosis_rag.loader import to_candidate
from kosis_rag.schemas import SearchResult, TableCandidate

NO_MATCH = "NO_MATCH"
REFUSAL_TEXT = "제공된 KOSIS 통계표 중에서 이 질문에 맞는 표를 찾지 못했습니다. 다른 표현으로 다시 질문해 주세요."
# Why the stage 2 LLM re-ranker picked nothing (kosis_rag.hcx.REASONS) -> the guide sentence; code picks it, not the LLM.
# Only out_of_scope gets its own sentence: on the T2-09 golden set HCX-007 labelled 9 of 13 out-of-scope refusals
# "no_item", so a "related tables exist" sentence would mislead; no_item falls back to REFUSAL_TEXT.
REFUSALS = {
    "out_of_scope": "이 질문은 제공된 KOSIS 인구 통계표가 다루는 범위 밖입니다. 다른 통계(예: 지역·경제·보건 통계)는 KOSIS에서 직접 찾아 주세요.",
}
# ponytail: no score threshold in the service yet (tau is calibrated only in evals); refusals are "no candidates" and
# the stage 2 LLM's empty pick (T2-09). A filter that removes every table refuses with the generic text.

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 KOSIS 통계표 안내 도우미입니다. 아래 후보 통계표 목록만 근거로, 질문에 맞는 표를 1~3개 골라 왜 맞는지 짧게 설명하세요.\n"
     "규칙: 목록에 없는 표·기간·숫자를 지어내지 마세요. 통계 수치는 말하지 마세요. 표를 가리킬 때는 [번호]와 표 ID를 쓰세요.\n"
     f"맞는 표가 하나도 없으면 다른 말 없이 {NO_MATCH} 만 출력하세요."),
    ("human", "후보 통계표:\n{context}\n\n질문: {question}"),
])

Route = Literal["refuse", "guide"]

_TBL_ID = re.compile(r"(?<![0-9A-Za-z_])DT_[0-9A-Za-z_]+")  # not \b: Korean letters count as \w ("표DT_X", T2-38)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def format_context(candidates: list[TableCandidate]) -> str:
    """Write the candidate tables as the numbered list the guide prompt shows the LLM.

    In: candidates = tables shown to the answer, in rank order. Out: one line per table,
    "[n] tbl_id | table name | survey | organization | period". Side effects: none. Node: guide.
    """
    return "\n".join(
        f"[{i}] {c.tbl_id} | {c.table_name} | {c.survey_name} | {c.organization} | {c.period}"
        for i, c in enumerate(candidates, 1)
    )


def clova_chat(settings=None) -> BaseChatModel:
    """HCX through langchain-naver's ChatClovaX; model and limits from configs/model.yaml.

    Key: HCX_API_KEY from this repo's .env (Phase 1 load_env() alone reads vendor/phase1/.env),
    else ChatClovaX's own CLOVASTUDIO_API_KEY.
    """
    from langchain_naver import ChatClovaX

    load_env(Path(__file__).resolve().parents[2] / ".env")
    settings = settings or load_settings()
    key = get_secret(settings.api_key_env)
    return ChatClovaX(
        model=settings.model_name, temperature=settings.temperature, max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds, **({"api_key": key} if key else {}),
        stream_usage=True,  # off by default for a non-OpenAI base URL: streamed chunks would carry no token usage (T2-38)
    )


def llm_runnable(client: LLMClient | BaseChatModel, on_chunk: Callable[[str], None] | None = None) -> Runnable:
    """Wrap a LangChain chat model or a Phase 1 client (mock) as a runnable: prompt messages in, LLMResult out.

    Does: lets the guide prompt call either model type the same way.
    In: client = a LangChain BaseChatModel (ChatClovaX) or a Phase 1 LLMClient (MockClient in tests);
        on_chunk = called with the reply so far after each streamed chunk (T2-38; chat models only).
    Out: a Runnable from a prompt value to an LLMResult (text, status, usage). A chat model is streamed and its
        chunks summed (text and usage_metadata); a model that cannot stream gives one chunk.
    Side effects: none when built; invoking it calls the model (network for real models). A chat model's exception,
        also one in the middle of the stream, is caught and returned as status STATUS_API_ERROR; a Phase 1 client
        reports failures in its own status (it gets no on_chunk calls).
    Node: guide (through guide_answer).
    """

    if isinstance(client, BaseChatModel):
        def chat(prompt_value) -> LLMResult:
            msg = None
            try:
                for chunk in client.stream(prompt_value):
                    msg = chunk if msg is None else msg + chunk
                    if on_chunk is not None:
                        on_chunk(msg.content)
            except Exception as e:  # keep the Phase 1 contract: a failed call is a status, not a raise
                return LLMResult(text="", status=STATUS_API_ERROR, error=f"{type(e).__name__}: {e}")
            if msg is None:
                return LLMResult(text="", status=STATUS_API_ERROR, error="empty stream")
            return LLMResult(text=msg.content, usage=dict(msg.usage_metadata or {}))

        return RunnableLambda(chat, name=f"chat_model:{client.get_name()}")

    def call(prompt_value) -> LLMResult:
        messages = prompt_value.to_messages()
        system = "\n".join(m.content for m in messages if m.type == "system")
        user = "\n".join(m.content for m in messages if m.type != "system")
        return client.generate(system, user)

    return RunnableLambda(call, name=f"phase1_llm:{client.name}")


def unsupported_claims(text: str, question: str, candidates: list[TableCandidate]) -> list[str]:
    """Table ids and years in the reply that neither the candidates nor the question contain.

    Does: checks the LLM's guide sentence for invented table ids or years before it is shown.
    In: text = the LLM reply; question = the user question (its years are allowed); candidates = the tables shown.
    Out: the unsupported ids and years, in the order found (empty = the reply passes).
    Side effects: none.
    Node: guide (through guide_answer).
    """
    ranges = [(min(ys), max(ys)) for c in candidates if (ys := [int(y) for y in _YEAR.findall(c.period)])]
    asked = set(_YEAR.findall(question))
    ids = [t for t in _TBL_ID.findall(text) if t not in {c.tbl_id for c in candidates}]
    years = [y for y in _YEAR.findall(text) if y not in asked and not any(lo <= int(y) <= hi for lo, hi in ranges)]
    return ids + years


def streamable_prefix(text: str, question: str, candidates: list[TableCandidate]) -> str | None:
    """The part of a reply still being written that may already be shown (T2-38).

    Does: lets the guide stream only what would pass the final check, so no invented id or year is ever on screen.
    In: text = the reply so far; question, candidates = as unsupported_claims.
    Out: the text up to its last whitespace (a cut word such as "DT_1B0" or "202" waits); "" while the reply may
        still be NO_MATCH (also "NO_MATCH\n"); None once that part names an unsupported id or year — the reply will
        fall back to the template, so nothing more is sent. Ids and years end at whitespace, so a claim found in a
        prefix is found in the whole reply: None stays None as the text grows.
    Side effects: none.
    Node: guide (through guide_answer).
    """
    head = text.lstrip()
    if NO_MATCH.startswith(head) or head.startswith(NO_MATCH):
        return ""
    words = re.match(r"[\s\S]*\s", text)
    prefix = words.group(0) if words else ""
    return None if unsupported_claims(prefix, question, candidates) else prefix


def template_answer(candidates: list[TableCandidate]) -> str:
    """The guide sentence written by code when the LLM failed or its reply did not pass verification.

    In: candidates = the tables shown, in rank order. Out: a fixed opening line and the top 3 tables (name, id,
    organization, period). Side effects: none. Node: guide (through guide_answer).
    """
    lines = ["질문과 관련 있어 보이는 통계표입니다."]
    lines += [f"[{i}] {c.table_name} ({c.tbl_id}) · {c.organization} · {c.period}" for i, c in enumerate(candidates[:3], 1)]
    return "\n".join(lines)


def with_citations(text: str, candidates: list[TableCandidate]) -> str:
    """Append the source lines (KOSIS / survey / table + link) of the top 3 candidates to the guide sentence.

    In: text = the guide sentence; candidates = the tables shown. Out: text + "출처:" block written by code.
    Side effects: none. Node: guide (through guide_answer).
    """
    return text + "\n\n출처:\n" + "\n".join(f"- {c.to_source().citation()}" for c in candidates[:3])


def search(retriever: BaseRetriever, question: str, k: int = 5) -> list[TableCandidate]:
    return [to_candidate(d) for d in retriever.invoke(question)[:k]]


def decide_route(docs: list[Document]) -> tuple[Route, str | None]:
    """Decide whether the answer refuses before calling the guide LLM.

    In: docs = the tables shown to the answer (already cut to k), in rank order, with the search metadata.
    Out: (route, refusal reason). "refuse" when there are no docs, or when stage 2 ran without error and picked
        nothing (T2-09; the reason is its stage2_reason, possibly None); otherwise ("guide", None).
        A stage 2 error is not a refusal: the answer keeps the stage 1 order.
    Side effects: none.
    Node: gate.
    """
    if not docs:
        return "refuse", None
    meta = docs[0].metadata
    if meta.get("stage2_picked_n") == 0 and not meta.get("stage2_error"):  # the LLM re-ranker picked nothing (T2-09)
        return "refuse", meta.get("stage2_reason")
    return "guide", None


def refusal_answer(question: str, docs: list[Document], reason: str | None) -> SearchResult:
    """The refusal answer: no table fits the question, said in a sentence code picks (no LLM call).

    In: question = the user question; docs = the tables shown (cut to k; empty when nothing was found);
        reason = why stage 2 picked nothing (kosis_rag.hcx.REASONS) or None.
    Out: SearchResult with no_answer=True and answer_origin="refusal". With no docs: no candidates and REFUSAL_TEXT.
        Otherwise the candidates are kept, no_answer_reason=reason and the sentence is REFUSALS[reason] or
        REFUSAL_TEXT.
    Side effects: none.
    Node: refuse.
    """
    if not docs:
        return SearchResult(question=question, no_answer=True, answer=REFUSAL_TEXT, answer_origin="refusal")
    return SearchResult(question=question, candidates=[to_candidate(d) for d in docs], no_answer=True,
                        no_answer_reason=reason, answer=REFUSALS.get(reason, REFUSAL_TEXT), answer_origin="refusal")


def guide_answer(question: str, docs: list[Document], client: LLMClient | BaseChatModel,
                 on_text: Callable[[str], None] | None = None) -> SearchResult:
    """The guide answer: the LLM explains which candidate tables fit, code verifies it and adds the citations.

    In: question = the user question; docs = the tables shown (cut to k, not empty), in rank order;
        client = the guide LLM (ChatClovaX or a Phase 1 client); on_text = called with each newly showable piece
        of the reply while it streams (streamable_prefix; T2-38), never with text the final check would reject.
    Out: SearchResult with the candidates. The LLM's NO_MATCH is a refusal (REFUSAL_TEXT); a failed call, an empty
        reply or a reply naming an id/year outside the candidates falls back to template_answer
        (answer_origin="template"); otherwise answer_origin="llm". Citations are appended by code.
    Side effects: one LLM call (network for a real model); model failures come back as a status, not an exception.
    Node: guide.
    """
    candidates = [to_candidate(d) for d in docs]
    sent = ""
    stopped = False

    def on_chunk(so_far: str) -> None:
        nonlocal sent, stopped
        prefix = None if stopped else streamable_prefix(so_far, question, candidates)
        if prefix is None:
            stopped = True
        elif len(prefix) > len(sent):
            on_text(prefix[len(sent):])
            sent = prefix

    # 1. ask the LLM for a guide sentence over the candidates (streamed piece by piece when on_text is given)
    result: LLMResult = (PROMPT | llm_runnable(client, on_chunk if on_text else None)).invoke(
        {"question": question, "context": format_context(candidates)}, config={"run_name": "table_guide_chain"}
    )
    text = result.text.strip()
    # 2. the LLM says nothing fits
    if result.status == STATUS_OK and text == NO_MATCH:
        return SearchResult(question=question, candidates=candidates, no_answer=True, answer=REFUSAL_TEXT,
                            answer_origin="refusal", llm_status=result.status)
    # 3. verify the reply; fall back to the code template
    origin = "llm"
    if result.status != STATUS_OK or not text or unsupported_claims(text, question, candidates):
        text, origin = template_answer(candidates), "template"
    return SearchResult(question=question, candidates=candidates, answer=with_citations(text, candidates),
                        answer_origin=origin, llm_status=result.status)


def answer(retriever: BaseRetriever, client: LLMClient | BaseChatModel, question: str, k: int = 5) -> SearchResult:
    """Search, then refuse or guide: the old (non-graph) answer path, same result as the graph.

    In: retriever = the staged search (build_search); client = the guide LLM; question = the user question;
        k = tables shown to the answer.
    Out: SearchResult from refusal_answer or guide_answer on the top k.
    Side effects: runs the retriever (Chroma, models, stage 2 LLM in production) and at most one guide LLM call.
    Node: not used by the graph (the graph runs decide_route / refusal_answer / guide_answer as gate / refuse / guide).
    """
    docs = retriever.invoke(question)[:k]
    route, reason = decide_route(docs)
    if route == "refuse":
        return refusal_answer(question, docs, reason)
    return guide_answer(question, docs, client)
