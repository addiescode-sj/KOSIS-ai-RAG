"""BM25 retriever that reports its score in Document.metadata["score"] (the no-answer check needs it)."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """Words plus Korean character bigrams, so '출생아수' also matches '출생아 수'."""
    # ponytail: no morphological analyzer; T2-04 compares a Korean tokenizer (e.g. Kiwi) against this.
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        tokens.append(word)
        tokens += [word[i : i + 2] for i in range(len(word) - 1)]
    return tokens


class BM25TableRetriever(BaseRetriever):
    """Top-k tables by BM25. Score 0 means no query token appears in any table text."""

    docs: list[Document]
    k: int = 20  # upper bound; callers slice to their own k (SearchArgs.k <= 20)
    bm25: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 20) -> BM25TableRetriever:
        return cls(docs=docs, k=k, bm25=BM25Okapi([tokenize(d.page_content) for d in docs]))

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [
            Document(page_content=self.docs[i].page_content, metadata={**self.docs[i].metadata, "score": float(scores[i])}, id=self.docs[i].id)
            for i in top
            if scores[i] > 0
        ]
