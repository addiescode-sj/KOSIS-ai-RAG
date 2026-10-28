"""Public API as LangChain tools: tracing (LangSmith / OpenTelemetry callbacks) and the eval harness both call these.

search_tables -> list[TableCandidate] (as dicts) is what T2-03 grades for Hit@k / MRR;
answer_table_question -> SearchResult adds the guide sentence and the no-answer decision.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool

from kosis_chat.llm.model import LLMClient

from kosis_rag import chain
from kosis_rag.schemas import SearchArgs


def build_tools(retriever: BaseRetriever, client: LLMClient) -> list[BaseTool]:
    """Bind the retriever and model once; swap the retriever (BM25 -> Chroma, T2-05) without touching the tools."""

    def search_tables(question: str, k: int = 5) -> list[dict]:
        return [c.model_dump() for c in chain.search(retriever, question, k)]

    def answer_table_question(question: str, k: int = 5) -> dict:
        return chain.answer(retriever, client, question, k).model_dump()

    return [
        StructuredTool.from_function(
            search_tables, name="search_tables", args_schema=SearchArgs,
            description="질문에 맞는 KOSIS 통계표 후보를 점수 순으로 돌려준다 (표 ID·기관·수록기간·링크).",
        ),
        StructuredTool.from_function(
            answer_table_question, name="answer_table_question", args_schema=SearchArgs,
            description="통계표 후보와 출처가 붙은 안내 문장을 돌려준다. 맞는 표가 없으면 no_answer=True.",
        ),
    ]
