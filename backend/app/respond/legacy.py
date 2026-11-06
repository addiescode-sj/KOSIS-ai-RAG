"""The old retriever-shaped answer (T2-41 S5: chain.answer and tools.py's answer_table_question): search with a
LangChain retriever, then refuse or guide — the same result as the answer graph, without LangGraph.

lab/rag_baseline.py and the tests call answer; answer_tool is the answer_table_question tool (the search_tables tool
is the module's, kosis_rag.legacy.tools.build_tools).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool

from kosis_rag.gate import node as gate_step
from kosis_rag.gate import texts
from kosis_rag.llm.base import LLMClient
from kosis_rag.schemas.io import SearchArgs, SearchResult
from kosis_rag.shared.catalog import to_candidate

from backend.app.respond.node import GuideInput, RefuseInput, guide, refuse


def answer(retriever: BaseRetriever, client: LLMClient | BaseChatModel, question: str, k: int = 5) -> SearchResult:
    """Search, then refuse or guide: the old (non-graph) answer path, same result as the answer graph.

    In: retriever = the staged search (kosis_rag.legacy.pipeline.build_search); client = the guide LLM;
        question = the user question; k = tables shown to the answer.
    Out: SearchResult from refuse or guide on the top k.
    Side effects: runs the retriever (Chroma, models, stage 2 LLM in production) and at most one guide LLM call.
    Node: not used by the graph (the graph runs gate.node.decide_route / refuse / guide as gate / refuse / guide).
    """
    docs = retriever.invoke(question)[:k]
    gate = gate_step.decide_route(gate_step.GateInput(docs=docs))
    candidates = [to_candidate(d) for d in docs]
    if gate.route == "refuse":
        return refuse(RefuseInput(question=question, candidates=candidates, refusal_reason=gate.refusal_reason,
                                  refusal_text=texts.REFUSALS.get(gate.refusal_reason, texts.REFUSAL_TEXT)))
    return guide(GuideInput(question=question, candidates=candidates), client)


def answer_tool(retriever: BaseRetriever, client: LLMClient | BaseChatModel) -> BaseTool:
    """The answer_table_question tool over a retriever and a guide LLM (T2-41 S5: from tools.build_tools).

    Out: a StructuredTool (SearchArgs in, SearchResult as a dict out). Side effects: none when built. Node: none.
    """

    def answer_table_question(question: str, k: int = 5) -> dict:
        return answer(retriever, client, question, k).model_dump()

    return StructuredTool.from_function(
        answer_table_question, name="answer_table_question", args_schema=SearchArgs,
        description="통계표 후보와 출처가 붙은 안내 문장을 돌려준다. 맞는 표가 없으면 no_answer=True.",
    )
