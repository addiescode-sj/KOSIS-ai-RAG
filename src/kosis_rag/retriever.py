"""BM25 and ChromaDB retrievers that report their score in Document.metadata["score"] (the no-answer check needs it)."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """Words plus Korean character bigrams, so '출생아수' also matches '출생아 수'."""
    # TODO(T2-04): placeholder tokenizer (no morphological analyzer). Replace with the Korean tokenizer T2-04 picks
    # (e.g. Kiwi); BM25 is rebuilt in memory, so changing this function is the whole swap.
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


class ChromaTableRetriever(BaseRetriever):
    """Top-k tables by cosine similarity over a T2-24 collection. Score = 1 - cosine distance (higher is closer).

    The collection holds vectors we computed ourselves (embedding_function=None), so the query must be embedded
    with the same model; `connect` refuses a collection built with another model.
    """

    collection: Any
    embed_query: Callable[[str], list[float]]
    k: int = 20  # upper bound; callers slice to their own k (SearchArgs.k <= 20)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def connect(cls, name: str | None = None, k: int = 20, device: str | None = None) -> ChromaTableRetriever:
        """Open CHROMA_COLLECTION (or `name`) on the shared server and load the query embedder."""
        from kosis_rag.index import EMBED_MODEL, kure_embedder
        from kosis_rag.vectorstore import chroma_client

        client = chroma_client()  # also loads .env
        name = name or os.getenv("CHROMA_COLLECTION")
        if not name:
            raise ValueError("Set CHROMA_COLLECTION in .env to the collection built by `python -m kosis_rag.index`")
        collection = client.get_collection(name, embedding_function=None)
        built_with = (collection.metadata or {}).get("embed_model")
        if built_with != EMBED_MODEL:
            raise ValueError(f"collection {name} was embedded with {built_with!r}, queries use {EMBED_MODEL!r}; reload it")
        # TODO(T2-05): query embedder = the same placeholder model/tokenizer as index.EMBED_MODEL (KURE-v1).
        # When T2-05 picks the final model, change it in index.py, reload a new collection and point CHROMA_COLLECTION at it.
        embed, _, _ = kure_embedder(device)
        return cls(collection=collection, embed_query=lambda text: embed([text])[0], k=k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        # TODO(T2-09): Chroma always returns k hits, so the chain's "no candidates" refusal never fires here;
        # refusal relies on the LLM's NO_MATCH until a cosine threshold is calibrated on the L1 dev split.
        res = self.collection.query(query_embeddings=[self.embed_query(query)], n_results=self.k,
                                    include=["documents", "metadatas", "distances"])
        return [
            Document(page_content=doc, metadata={**meta, "score": 1.0 - dist}, id=key)
            for key, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
