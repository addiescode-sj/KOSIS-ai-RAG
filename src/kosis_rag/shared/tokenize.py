"""The BM25 tokenizer (T2-41 S2: from retriever.py). Shared by the table BM25, the question BM25 and lab."""

from __future__ import annotations

import re

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
