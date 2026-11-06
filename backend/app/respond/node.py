"""The app's answer step functions and their graph nodes (T2-41 S5: chain.refusal_answer / guide_answer and the
module graph's refuse / guide nodes).

    refuse(RefuseInput) -> SearchResult                  code sentence from the module's SearchOutput, no LLM
    guide(GuideInput, llm, on_text) -> SearchResult      guide LLM over the candidates, checked, cited by code
    answer_nodes(guide_client, k=k)                      the graph nodes refuse / guide over the API state

The guide LLM is GuideClient: made from its LLMConfig (llm.providers.build_client) when guide first runs and kept
(the backend's Runtime holds one), so parse ... gate and refuse run without a guide key, and a missing key is
LLMKeyMissing at guide's entry.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.config import get_stream_writer
from pydantic import Field

from kosis_rag.api import to_output
from kosis_rag.gate import texts
from kosis_rag.graph.spans import timed
from kosis_rag.llm.base import STATUS_API_ERROR, STATUS_OK, LLMClient, LLMResult
from kosis_rag.llm.config import LLMConfig, LLMKeyMissing
from kosis_rag.llm.providers import build_client
from kosis_rag.schemas.io import SearchResult, TableCandidate
from kosis_rag.shared.boundary import BoundaryModel
from kosis_rag.shared.catalog import to_candidate

from backend.app.respond.prompts import guide_prompt
from backend.app.respond.render import format_context, template_answer, with_citations
from backend.app.respond.state import SearchState
from backend.app.respond.verify import picked_from_reply, streamable_prefix, unsupported_claims


class RefuseInput(BoundaryModel):
    """In of refuse: the module's SearchOutput fields the refusal answer reads."""

    question: str
    candidates: list[TableCandidate] = Field(default_factory=list)  # SearchOutput.candidates (docs[:k])
    refusal_reason: str | None = None  # SearchOutput.refusal_reason
    refusal_text: str  # SearchOutput.refusal_text (gate.texts REFUSALS.get(reason, REFUSAL_TEXT))


class GuideInput(BoundaryModel):
    """In of guide: the question and the tables shown to the guide LLM (docs[:k], not empty), in rank order."""

    question: str
    candidates: list[TableCandidate]


def refuse(inp: RefuseInput) -> SearchResult:
    """The refusal answer: no table fits the question, said in the sentence the module chose (no LLM call).

    In: inp = the module's output on refuse. Out: SearchResult with no_answer=True, answer_origin="refusal",
        answer = refusal_text, the candidates kept and no_answer_reason = refusal_reason (both empty / None when
        the search had no table). Side effects: none. Node: refuse.
    """
    return SearchResult(question=inp.question, candidates=inp.candidates, no_answer=True,
                        no_answer_reason=inp.refusal_reason, answer=inp.refusal_text, answer_origin="refusal")


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
    Node: guide (through guide).
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


def guide(inp: GuideInput, llm: LLMClient | BaseChatModel,
          on_text: Callable[[str], None] | None = None) -> SearchResult:
    """The guide answer: the LLM explains which candidate tables fit, code verifies it and adds the citations.

    In: inp = the question and the tables shown (not empty), in rank order; llm = the guide LLM (ChatClovaX or a
        Phase 1 client); on_text = called with each newly showable piece of the reply while it streams
        (streamable_prefix; T2-38), never with text the final check would reject.
    Out: SearchResult with the candidates. The LLM's NO_MATCH is a refusal (REFUSAL_TEXT, picked=[]); a failed call,
        an empty reply, a reply naming an id/year/[n] outside the candidates or naming no table at all falls back to
        template_answer (answer_origin="template", picked = its top 3); otherwise answer_origin="llm" and picked =
        the tables the reply names (picked_from_reply). Citations are appended by code for exactly the picked tables.
    Side effects: one LLM call (network for a real model); model failures come back as a status, not an exception.
    Node: guide.
    """
    question, candidates = inp.question, inp.candidates
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
    result: LLMResult = (guide_prompt() | llm_runnable(llm, on_chunk if on_text else None)).invoke(
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


class GuideClient:
    """The guide LLM client, made on first use and kept (T2-41 S5; the S4 factory path in the module graph is gone).

    In: llm = an LLMConfig (built through llm.providers.build_client when guide first runs; provider clova_chat in
        the backend), a ready client (LLMClient | BaseChatModel: tests, MockClient), or None (no guide LLM).
    """

    def __init__(self, llm: LLMConfig | LLMClient | BaseChatModel | None) -> None:
        self.config = llm if isinstance(llm, LLMConfig) or llm is None else None
        self._client: Any = None if isinstance(llm, LLMConfig) or llm is None else llm
        self._lock = threading.Lock()

    def get(self) -> Any:
        """The client. Side effects: the first call with a config builds it (one build even when guide runs
        concurrently); LLMKeyMissing when there is no config or its api_key is None (nothing is kept, the next call
        checks again)."""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                if self.config is None or self.config.api_key is None:
                    raise LLMKeyMissing("guide LLM key not set (HCX_API_KEY or CLOVASTUDIO_API_KEY in the environment)")
                self._client = build_client(self.config)
            return self._client


def as_guide_client(llm: GuideClient | LLMConfig | LLMClient | BaseChatModel | None) -> GuideClient:
    """llm as a GuideClient (a GuideClient is returned as it is, so its cache is shared)."""
    return llm if isinstance(llm, GuideClient) else GuideClient(llm)


Node = Callable[[SearchState], dict]


def answer_nodes(guide_client: GuideClient, *, k: int = 5) -> dict[str, Node]:
    """The graph's answer nodes over the API state: refuse and guide on the top k.

    In: guide_client = the guide LLM (GuideClient); k = the candidate budget (as the search nodes').
    Out: {"refuse": fn, "guide": fn}, fn(SearchState) -> {"result": SearchResult, "trace"}.
    Side effects: none when made. Node: none (it makes them).
    """

    def refuse_node(state: SearchState) -> dict:
        """Build the refusal answer on the top k from the module's output (kosis_rag.api.to_output).

        In: state.question, state.docs, state.refusal_reason. Out: {"result": SearchResult, "trace"}.
        Side effects: none (no LLM call). Node: refuse.
        """
        started = time.perf_counter()
        if not state.docs[:k]:  # no table: no reason, as gate gives none (a node API request may send one)
            state = state.model_copy(update={"refusal_reason": None})
        out = to_output(state, k)
        result = refuse(RefuseInput(question=out.question, candidates=out.candidates,
                                    refusal_reason=out.refusal_reason, refusal_text=out.refusal_text))
        return {"result": result, "trace": timed("refuse", state.trace, started)}

    def guide_node(state: SearchState) -> dict:
        """Build the guide answer on the top k.

        In: state.question, state.docs. Out: {"result": SearchResult, "trace"}; under graph.stream with "custom" mode
            also {"delta": text} pieces of the reply that already pass the check (verify.streamable_prefix, T2-38).
        Side effects: on the first run the guide client is built (LLMKeyMissing without a key); one guide LLM call
            (failures come back as a status, then the code template).
        Node: guide.
        """
        started = time.perf_counter()
        llm = guide_client.get()
        try:  # graph.stream(stream_mode=[..., "custom"]) receives the pieces; invoke ignores them
            writer = get_stream_writer()
        except RuntimeError:  # run outside a graph (the node API's /nodes/answer)
            writer = None
        on_text = (lambda piece: writer({"delta": piece})) if writer else None
        inp = GuideInput(question=state.question, candidates=[to_candidate(d) for d in state.docs[:k]])
        result = guide(inp, llm, on_text=on_text)
        return {"result": result, "trace": timed("guide", state.trace, started, origin=result.answer_origin)}

    return {"refuse": refuse_node, "guide": guide_node}
