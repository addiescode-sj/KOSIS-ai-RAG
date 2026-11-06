"""The search as a LangChain tool (T2-41 S5: from tools.py): tracing (LangSmith / OpenTelemetry callbacks) and the
eval harness call it.

search_tables -> list[TableCandidate] (as dicts) is what T2-03 grades for Hit@k / MRR. The answer tool
(answer_table_question: the guide sentence and the no-answer decision) is the app's: backend.app.respond.legacy.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool

from kosis_rag.legacy.search import search
from kosis_rag.schemas.io import SearchArgs


def build_tools(retriever: BaseRetriever) -> list[BaseTool]:
    """Bind the retriever once; swap the retriever (BM25 -> Chroma, T2-05) without touching the tool.

    Out: [search_tables]. Side effects: none when built. Node: none.
    """

    def search_tables(question: str, k: int = 5) -> list[dict]:
        return [c.model_dump() for c in search(retriever, question, k)]

    return [
        StructuredTool.from_function(
            search_tables, name="search_tables", args_schema=SearchArgs,
            description="질문에 맞는 KOSIS 통계표 후보를 점수 순으로 돌려준다 (표 ID·기관·수록기간·링크).",
        ),
    ]
