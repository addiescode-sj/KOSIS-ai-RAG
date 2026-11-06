"""RAG answer: retrieve candidate tables -> prompt -> LLM (ChatClovaX or an LLMClient) -> verify -> add citations by code.

The model only writes a guide sentence over the candidates. Table ids, periods and citations come from the catalog;
a reply that names an id or year not in the candidates, or a failed model call, falls back to a code template.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from kosis_rag.gate import node as gate_step  # module access: chain re-exports none of gate's names (T2-41 S3)
from kosis_rag.gate import texts
from kosis_rag.llm import providers
from kosis_rag.llm.base import STATUS_API_ERROR, STATUS_OK, LLMClient, LLMResult
from kosis_rag.llm.config import LLMConfig
from kosis_rag.schemas.io import SearchResult, TableCandidate
from kosis_rag.shared.catalog import to_candidate

# The guide model's settings: Phase 1 configs/model.yaml @e248a45, which clova_chat read through Phase 1 until
# T2-41 S3 (tests/test_phase1_contract.py checks they still match). S4 moves them to backend/app/config.py.
GUIDE_MODEL = "HCX-DASH-002"
GUIDE_TEMPERATURE = 0.0
GUIDE_MAX_TOKENS = 1024
GUIDE_TIMEOUT = 60.0
GUIDE_KEY_ENV = "HCX_API_KEY"

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 KOSIS 통계표 안내 도우미입니다. 아래 후보 통계표 목록만 근거로, 질문에 맞는 표를 1~3개 골라 왜 맞는지 짧게 설명하세요.\n"
     "규칙: 목록에 없는 표·기간·숫자를 지어내지 마세요. 통계 수치는 말하지 마세요. 표를 가리킬 때는 [번호]와 표 ID를 쓰세요.\n"
     "고른 표만 [번호]와 표 ID로 가리키고, 고르지 않은 표는 언급하지 마세요.\n"
     f"맞는 표가 하나도 없으면 다른 말 없이 {texts.NO_MATCH} 만 출력하세요."),
    ("human", "후보 통계표:\n{context}\n\n질문: {question}"),
])

_TBL_ID = re.compile(r"(?<![0-9A-Za-z_])DT_[0-9A-Za-z_]+")  # not \b: Korean letters count as \w ("표DT_X", T2-38)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# "[1]", "[1, 2]", "[1~3]", full-width "［1］"; "[1]~[3]" is read as a range too (T2-39)
_REF_RANGE = re.compile(r"[\[［]\s*(\d{1,3})\s*[\]］]\s*[~\-]\s*[\[［]\s*(\d{1,3})\s*[\]］]")
_REF = re.compile(r"[\[［]\s*(\d{1,3}(?:\s*[,，~\-]\s*\d{1,3})*)\s*[\]］]")


def format_context(candidates: list[TableCandidate]) -> str:
    """Write the candidate tables as the numbered list the guide prompt shows the LLM.

    In: candidates = tables shown to the answer, in rank order. Out: one line per table,
    "[n] tbl_id | table name | survey | organization | period". Side effects: none. Node: guide.
    """
    return "\n".join(
        f"[{i}] {c.tbl_id} | {c.table_name} | {c.survey_name} | {c.organization} | {c.period}"
        for i, c in enumerate(candidates, 1)
    )


def clova_chat(config: LLMConfig | None = None) -> BaseChatModel:
    """HCX through langchain-naver's ChatClovaX (llm.providers.clova_chat): the guide LLM.

    In: config = provider clova_chat settings; None = GUIDE_MODEL, GUIDE_TEMPERATURE, GUIDE_MAX_TOKENS and
        GUIDE_TIMEOUT (Phase 1 configs/model.yaml) with HCX_API_KEY from this repo's .env (T2-41 S3: still read here;
        S4 moves it to backend/app/config.py), else ChatClovaX's own CLOVASTUDIO_API_KEY.
    Out: the chat model. Side effects: with no config, loads this repo's .env into the environment (existing
        variables kept). Node: guide (built once by the backend).
    """
    if config is None:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        config = LLMConfig(provider="clova_chat", api_key=os.environ.get(GUIDE_KEY_ENV) or None, model=GUIDE_MODEL,
                           temperature=GUIDE_TEMPERATURE, max_tokens=GUIDE_MAX_TOKENS, timeout=GUIDE_TIMEOUT)
    return providers.clova_chat(config)


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
    Out: the unsupported ids and years, then [n] numbers outside the candidate list, in the order found
        (empty = the reply passes).
    Side effects: none.
    Node: guide (through guide_answer).
    """
    ranges = [(min(ys), max(ys)) for c in candidates if (ys := [int(y) for y in _YEAR.findall(c.period)])]
    asked = set(_YEAR.findall(question))
    ids = [t for t in _TBL_ID.findall(text) if t not in {c.tbl_id for c in candidates}]
    years = [y for y in _YEAR.findall(text) if y not in asked and not any(lo <= int(y) <= hi for lo, hi in ranges)]
    numbers = [f"[{n}]" for _, n in _references(text) if not 1 <= n <= len(candidates)]  # e.g. [7] of 5 (T2-39)
    return ids + years + numbers


def _references(text: str) -> list[tuple[int, int]]:
    """(position, number) of every [n] reference, ranges expanded ("[1~3]", "[1]~[3]" -> 1, 2, 3)."""

    def span(lo: int, hi: int) -> list[int]:
        return list(range(lo, hi + 1)) if lo <= hi <= lo + 20 else [lo, hi]  # a reversed or huge "range" is two refs

    refs: list[tuple[int, int]] = []
    taken: list[tuple[int, int]] = []
    for m in _REF_RANGE.finditer(text):
        refs += [(m.start(), n) for n in span(int(m.group(1)), int(m.group(2)))]
        taken.append(m.span())
    for m in _REF.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        for part in re.split(r"\s*[,，]\s*", m.group(1)):
            bounds = [int(x) for x in re.split(r"\s*[~\-]\s*", part)]
            refs += [(m.start(), n) for n in span(bounds[0], bounds[-1])]
    return sorted(refs)


def picked_from_reply(text: str, candidates: list[TableCandidate]) -> list[TableCandidate]:
    """The candidate tables a checked guide reply points at, in the order it first names them (T2-39).

    Does: turns the LLM's choice into data, so citations and the shown tables are the ones the reply recommends.
    In: text = a reply that passed unsupported_claims; candidates = the tables shown to the LLM, in rank order ([n] = n-th).
    Out: candidates named by table id (_TBL_ID, so "표DT_X" counts) or by [n], without repeats. A [n] right before an id
        ("[2] DT_A") defers to the id when they disagree: the id is the data. Empty when the reply names no table.
    Side effects: none.
    Node: guide (through guide_answer).
    """
    by_id = {c.tbl_id: c for c in candidates}
    ids = [(m.start(), m.end(), m.group(0)) for m in _TBL_ID.finditer(text)]
    hits: list[tuple[int, TableCandidate]] = [(start, by_id[t]) for start, _, t in ids if t in by_id]
    for pos, n in _references(text):
        if not 1 <= n <= len(candidates):
            continue
        close = text.find("]", pos)
        close = text.find("］", pos) if close == -1 else close
        if any(0 <= start - close <= 3 for start, _, _ in ids):  # "[2] DT_A": the id decides
            continue
        hits.append((pos, candidates[n - 1]))
    picked: list[TableCandidate] = []
    for _, c in sorted(hits, key=lambda h: h[0]):
        if c not in picked:
            picked.append(c)
    return picked


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
    if texts.NO_MATCH.startswith(head) or head.startswith(texts.NO_MATCH):
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


def with_citations(text: str, tables: list[TableCandidate]) -> str:
    """Append the source lines (KOSIS / survey / table + link) of the tables the sentence recommends.

    In: text = the guide sentence; tables = the tables it points at (picked_from_reply, or the template's top 3).
    Out: text + "출처:" block written by code. Side effects: none. Node: guide (through guide_answer).
    """
    return text + "\n\n출처:\n" + "\n".join(f"- {c.to_source().citation()}" for c in tables)


def search(retriever: BaseRetriever, question: str, k: int = 5) -> list[TableCandidate]:
    return [to_candidate(d) for d in retriever.invoke(question)[:k]]


def refusal_answer(question: str, docs: list[Document], reason: str | None) -> SearchResult:
    """The refusal answer: no table fits the question, said in a sentence code picks (no LLM call).

    In: question = the user question; docs = the tables shown (cut to k; empty when nothing was found);
        reason = why stage 2 picked nothing (kosis_rag.rerank.hcx.REASONS) or None.
    Out: SearchResult with no_answer=True and answer_origin="refusal". With no docs: no candidates and REFUSAL_TEXT.
        Otherwise the candidates are kept, no_answer_reason=reason and the sentence is REFUSALS[reason] or
        REFUSAL_TEXT.
    Side effects: none.
    Node: refuse.
    """
    if not docs:
        return SearchResult(question=question, no_answer=True, answer=texts.REFUSAL_TEXT, answer_origin="refusal")
    return SearchResult(question=question, candidates=[to_candidate(d) for d in docs], no_answer=True,
                        no_answer_reason=reason, answer=texts.REFUSALS.get(reason, texts.REFUSAL_TEXT),
                        answer_origin="refusal")


def guide_answer(question: str, docs: list[Document], client: LLMClient | BaseChatModel,
                 on_text: Callable[[str], None] | None = None) -> SearchResult:
    """The guide answer: the LLM explains which candidate tables fit, code verifies it and adds the citations.

    In: question = the user question; docs = the tables shown (cut to k, not empty), in rank order;
        client = the guide LLM (ChatClovaX or a Phase 1 client); on_text = called with each newly showable piece
        of the reply while it streams (streamable_prefix; T2-38), never with text the final check would reject.
    Out: SearchResult with the candidates. The LLM's NO_MATCH is a refusal (REFUSAL_TEXT, picked=[]); a failed call,
        an empty reply, a reply naming an id/year/[n] outside the candidates or naming no table at all falls back to
        template_answer (answer_origin="template", picked = its top 3); otherwise answer_origin="llm" and picked =
        the tables the reply names (picked_from_reply). Citations are appended by code for exactly the picked tables.
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
    if result.status == STATUS_OK and text == texts.NO_MATCH:
        return SearchResult(question=question, candidates=candidates, no_answer=True, answer=texts.REFUSAL_TEXT,
                            answer_origin="refusal", llm_status=result.status)
    # 3. verify the reply and read which tables it recommends; fall back to the code template (its top 3)
    picked = picked_from_reply(text, candidates) if result.status == STATUS_OK and text else []
    origin = "llm"
    if not picked or unsupported_claims(text, question, candidates):  # a reply naming no table cannot be cited
        text, origin, picked = template_answer(candidates), "template", candidates[:3]
    return SearchResult(question=question, candidates=candidates, picked=picked, answer=with_citations(text, picked),
                        answer_origin=origin, llm_status=result.status)


def answer(retriever: BaseRetriever, client: LLMClient | BaseChatModel, question: str, k: int = 5) -> SearchResult:
    """Search, then refuse or guide: the old (non-graph) answer path, same result as the graph.

    In: retriever = the staged search (build_search); client = the guide LLM; question = the user question;
        k = tables shown to the answer.
    Out: SearchResult from refusal_answer or guide_answer on the top k.
    Side effects: runs the retriever (Chroma, models, stage 2 LLM in production) and at most one guide LLM call.
    Node: not used by the graph (the graph runs gate.node.decide_route / refusal_answer / guide_answer as gate / refuse /
        guide).
    """
    docs = retriever.invoke(question)[:k]
    gate = gate_step.decide_route(gate_step.GateInput(docs=docs))
    if gate.route == "refuse":
        return refusal_answer(question, docs, gate.refusal_reason)
    return guide_answer(question, docs, client)
