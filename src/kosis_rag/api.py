"""The search module's entry point (T2-41 S4): SearchModule(spec, options, llm).search(SearchInput) -> SearchOutput.

    module = SearchModule(IndexSpec(chroma=ChromaConfig(collection="kosis_a_bgeft_..."), embed_ckpt="data/models/..."),
                          SearchOptions(stage2="none"))
    out = module.search(SearchInput(question="2023년 시도별 출생아 수"))
    out.route, out.picked[0].key   # "found", "101_DT_..." (the key Phase 1 summarize_stat looks up, not tbl_id)

The module opens its resources once (adapters.loader.load_resources: records, Chroma, models, question index) and
compiles one graph per k. It reads no .env and no environment: the caller declares the settings (§0.6).

Until S5 the graph still ends in the app's answer nodes (refuse | guide, plans/T2-41.md); the module runs them with
a mock guide client (no LLM call) and reads only what gate decided: to_output projects the final state to
SearchOutput by the rules of tests/parity_cases.project_output (the S0 fixtures' output.json).
"""

from __future__ import annotations

from typing import Any, Iterator

from kosis_rag.adapters.loader import load_resources
from kosis_rag.gate.texts import REFUSAL_TEXT, REFUSALS
from kosis_rag.graph.search import build_graph, run_search, stream_search
from kosis_rag.graph.spans import spans_of
from kosis_rag.llm.base import MockClient
from kosis_rag.llm.config import LLMConfig
from kosis_rag.schemas.io import SearchInput, SearchOutput, TableCandidate
from kosis_rag.schemas.resources import IndexSpec, SearchOptions, SearchResources
from kosis_rag.schemas.state import SearchState
from kosis_rag.shared.catalog import to_candidate

APP_NODES = ("refuse", "guide")  # the app's answer nodes, still in the graph until S5: not module spans


def to_output(state: SearchState, k: int) -> SearchOutput:
    """The module's SearchOutput read off a final graph state (T2-41 S4; the rules of parity_cases.project_output).

    In: state = run_search(...)'s final state; k = the candidate budget the graph was built with.
    Out: route found (gate's guide) | refuse; refusal_text = REFUSALS.get(reason, REFUSAL_TEXT) on refuse, else None;
        candidates = to_candidate(docs[:k]); picked = [] on refuse, the docs stage 2 picked (rank order) when it ran
        without error and picked something, else the candidates; spans = the trace in node order without the app's
        refuse / guide nodes.
    Side effects: none. Node: none (SearchModule.search / stream call it).
    """
    refuse = state.route == "refuse"
    docs = state.docs[:k]
    candidates = [to_candidate(d) for d in docs]
    top = docs[0].metadata if docs else {}
    if refuse:
        picked: list[TableCandidate] = []
    elif (top.get("stage2_picked_n") or 0) > 0 and not top.get("stage2_error"):
        picked = [to_candidate(d) for d in state.docs if d.metadata.get("stage2_picked")]
    else:
        picked = candidates
    return SearchOutput(
        question=state.question,
        route="refuse" if refuse else "found",
        refusal_reason=state.refusal_reason,
        refusal_text=REFUSALS.get(state.refusal_reason, REFUSAL_TEXT) if refuse else None,
        picked=picked,
        candidates=candidates,
        spans=spans_of(state.trace, skip=APP_NODES),
    )


class SearchModule:
    """Table search over one setup: open once, then search many questions (T2-41 S4)."""

    def __init__(self, spec: IndexSpec, options: SearchOptions | None = None, llm: LLMConfig | None = None) -> None:
        """Open the resources (records, Chroma, models, question index) for spec / options / the stage 2 LLM.

        Side effects: those of adapters.loader.load_resources (file reads, Chroma connection, model loads).
        """
        self._init(load_resources(spec=spec, options=options or SearchOptions(), llm=llm))

    def _init(self, resources: SearchResources) -> None:
        self.resources = resources
        self._graphs: dict[int, Any] = {}

    @classmethod
    def from_resources(cls, resources: SearchResources) -> SearchModule:
        """A module over resources already loaded (tests with fakes, the backend). Side effects: none."""
        module = cls.__new__(cls)
        module._init(resources)
        return module

    def k(self, requested: int | None) -> int:
        """The candidate budget: the request's k, else options.answer_k."""
        return requested or self.resources.options.answer_k

    def graph(self, k: int):
        """The compiled search graph for budget k (one per k, compiled on first use)."""
        if k not in self._graphs:
            self._graphs[k] = build_graph(self.resources, MockClient(""), k=k)  # guide: no LLM call (until S5)
        return self._graphs[k]

    def search(self, inp: SearchInput) -> SearchOutput:
        """Run the search for one question. Side effects: the searches and models the resources hold."""
        k = self.k(inp.k)
        return to_output(run_search(self.graph(k), inp.question), k)

    def stream(self, inp: SearchInput) -> Iterator[tuple[str, Any]]:
        """Run the search and yield ("<node>", its trace entry) per finished search node, then ("output", SearchOutput)
        — the same value search() returns. The app's refuse / guide events and tokens are not sent."""
        k = self.k(inp.k)
        for name, payload in stream_search(self.graph(k), inp.question):
            if name == "result":
                yield "output", to_output(payload, k)
            elif name != "token" and name not in APP_NODES:
                yield name, payload

    def search_tables(self, question: str, k: int | None = None) -> list[TableCandidate]:
        """The candidate tables for a question (= search(...).candidates)."""
        return self.search(SearchInput(question=question, k=k)).candidates
