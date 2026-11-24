"""Answerability stage step function (KOS-193): the classifier's verdict on the stage 1 top, before stage 2.

`resources` is the search resources (schemas.resources.SearchResources): answer_score and options.answer_top /
answer_theta. The verdict is written on the docs' metadata (answer_refused, answer_top_score on every doc;
answer_score on each scored doc), where rerank2 (skip the LLM) and gate (refuse with "not_answerable") read it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from kosis_rag.answerable.schemas import AnswerInput, AnswerOutput

AnswerScore = Callable[[str, list[str]], list[float]]  # (question, table keys) -> one classifier logit per key


def card_scorer(scorer: Any, card_of: dict[str, str]) -> AnswerScore:
    """Score function over table keys from a loaded classifier and the cards it was trained on.

    In: scorer = adapters.models.AnswerScorer (or any object with .score(questions, cards)); card_of = table key ->
        card text (answerable.cards.load_cards). A key without a card is scored on the key text itself, as
        scripts/answerability/score_run.py does.
    Out: score(question, keys) -> one raw logit per key, in key order (the question as given, no query rules).
    Side effects: none when built; each call runs the classifier (GPU/CPU inference).
    Node: answerable (built by load_resources, called inside judge).
    """

    def score(question: str, keys: list[str]) -> list[float]:
        return scorer.score([question] * len(keys), [card_of.get(k, k) for k in keys])

    return score


def judge(inp: AnswerInput, resources: Any) -> AnswerOutput:
    """Score the first answer_top docs and mark the question refused when even the best is below answer_theta.

    In: inp.question, inp.docs (current order, metadata["key"] set); resources.answer_score, options.answer_top and
        options.answer_theta (a float: load_resources requires it with the classifier on).
    Out: AnswerOutput: new Document objects in the same order (the inputs are not changed); every doc gets
        answer_top_score and answer_refused, each scored doc also answer_score (its own logit). refused = top_score
        < answer_theta (strict). No docs: docs [], top_score None, refused False, n_scored 0 (gate refuses an empty
        list on its own).
    Side effects: one answer_score call (classifier inference) when there are docs. Node: answerable.
    """
    if not inp.docs:
        return AnswerOutput(docs=[], top_score=None, refused=False, n_scored=0)
    o = resources.options
    head = inp.docs[:o.answer_top]
    scores = [float(s) for s in resources.answer_score(inp.question, [d.metadata["key"] for d in head])]
    top = max(scores) if scores else None
    refused = top is not None and top < o.answer_theta
    verdict = {"answer_top_score": top, "answer_refused": refused}
    docs = [Document(page_content=d.page_content, id=d.id,
                     metadata={**d.metadata, **verdict, **({"answer_score": scores[i]} if i < len(scores) else {})})
            for i, d in enumerate(inp.docs)]
    return AnswerOutput(docs=docs, top_score=top, refused=refused, n_scored=len(scores))
