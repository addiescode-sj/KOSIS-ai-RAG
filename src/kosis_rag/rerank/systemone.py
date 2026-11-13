"""A Jev-style decision model as the stage 2 re-ranker (T3-23 Jev: Clef-Flash or K-Decision-9B over /v1/systemone).

The model returns probabilities, not a list, so the selector ranks and cuts in code:
    noul    one true/false question per table ("does this table hold the measure the question asks?"), all k_in tables
            in one request; a table is picked when its probability >= threshold, highest first, at most MAX_PICK.
    choice  one choice question whose options are the tables; the same cut on the option probabilities.
Nothing at or above the threshold is the no-answer verdict: keys [] (rerank_stage2 records stage2_picked_n = 0, the
gate refuses), reason None. No "none of these" option is offered (it empties too many answers on Jev-style models).
`scores` (key -> probability, in input order) goes to metadata["stage2_scores"], so a threshold can be re-chosen
offline without calling the model again.

The wording is package data, rerank/prompts/stage2-jev-<mode>-<version>.md (jev_prompt), in blocks:
    <!-- state -->         the request's state; {question} = the user question (no block: the question as is)
    <!-- instructions -->  noul: one table's question, {table} = its card text (required); choice: the one question
    <!-- true --> <!-- false -->  noul only, optional: sent as the question's criteria {"true", "false"}
Text before the first block is a note and is not sent. Placeholders are replaced as plain text.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from kosis_rag.llm.config import LLMConfig, LLMKeyMissing
from kosis_rag.llm.systemone import SystemOneClient
from kosis_rag.rerank.constants import JEV_MODES, JEV_PROMPT_VERSION, JEV_THRESHOLD, MAX_PICK


@dataclass
class DecisionResult:
    keys: list[str]  # picked tables, most probable first
    calls: int = 1
    usage: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    scores: dict[str, float] = field(default_factory=dict)  # every table sent -> its probability


def cut(scores: dict[str, float], threshold: float, max_pick: int = MAX_PICK) -> list[str]:
    """Keys with probability >= threshold, highest first (ties keep input order), at most max_pick."""
    ranked = sorted(scores, key=lambda k: -scores[k])  # sorted() is stable: ties keep input order
    return [k for k in ranked if scores[k] >= threshold][:max_pick]


BLOCK = re.compile(r"<!--\s*(state|instructions|true|false)\s*-->")


@dataclass(frozen=True)
class JevPrompt:
    state: str  # with {question}
    instructions: str  # noul: with {table}
    true: str | None = None
    false: str | None = None


@functools.cache
def jev_prompt(mode: str, version: str = JEV_PROMPT_VERSION) -> JevPrompt:
    """rerank/prompts/stage2-jev-<mode>-<version>.md parsed into blocks (read once).

    Side effects: reads the package data file (FileNotFoundError when missing); ValueError when instructions is
    missing, or a noul prompt has no {table} / a choice prompt has one. Node: rerank2.
    """
    name = f"stage2-jev-{mode}-{version}.md"
    text = (resources.files("kosis_rag.rerank") / "prompts" / name).read_text(encoding="utf-8")
    parts = BLOCK.split(text.replace("\r\n", "\n"))
    blocks = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if not blocks.get("instructions"):
        raise ValueError(f"{name}: no <!-- instructions --> block")
    if (mode == "noul") != ("{table}" in blocks["instructions"]):
        raise ValueError(f"{name}: {{table}} must be in noul instructions and only there")
    return JevPrompt(state=blocks.get("state") or "{question}", instructions=blocks["instructions"],
                     true=blocks.get("true") or None, false=blocks.get("false") or None)


def noul_questions(passages: list[str], prompt: JevPrompt) -> dict[str, dict]:
    criteria = ({"criteria": {"true": prompt.true or "", "false": prompt.false or ""}}
                if prompt.true or prompt.false else {})
    return {f"t{n}": {"type": "noul", "instructions": prompt.instructions.replace("{table}", p), **criteria}
            for n, p in enumerate(passages, 1)}


def choice_questions(passages: list[str], prompt: JevPrompt) -> dict[str, dict]:
    return {"pick": {"type": "choice", "instructions": prompt.instructions,
                     "criteria": {f"t{n}": p for n, p in enumerate(passages, 1)}}}


NOUL_KEYS = ("noul", "probability", "p", "yes", "true")  # Clef: {"noul": p}; other Jev-style servers (K-Decision)


def noul_probability(answer: Any) -> float:
    """One noul answer -> P(true): a number, or the first numeric NOUL_KEYS field; 0.0 when there is none."""
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, dict):
        for k in NOUL_KEYS:
            if isinstance(answer.get(k), (int, float)):
                return float(answer[k])
    return 0.0


def probabilities(reply: dict, mode: str, n: int) -> list[float]:
    """Reply -> one probability per table t1..tn (0.0 for a table the reply lacks). Raises ValueError without answers."""
    answers = reply.get("answers")
    if not isinstance(answers, dict):
        raise ValueError(f"no answers in reply: {str(reply)[:80]!r}")
    if mode == "noul":
        return [noul_probability(answers.get(f"t{i}")) for i in range(1, n + 1)]
    probs = (answers.get("pick") or {}).get("probabilities") or {}
    if isinstance(probs, list):  # a list in option order
        probs = {f"t{i}": p for i, p in enumerate(probs, 1)}
    return [float(probs.get(f"t{i}") or 0.0) for i in range(1, n + 1)]


@dataclass
class DecisionSelector:
    """select(question, keys) -> DecisionResult. `text_of` maps a table key to the passage sent for it.

    mode / threshold / prompt_version: None = config's decision_mode / threshold / prompt_version, else noul,
    JEV_THRESHOLD[mode] and JEV_PROMPT_VERSION.
    client: the systemone client (tests pass a fake); None = built from `config` at the first call.
    """

    text_of: dict[str, str]
    config: LLMConfig | None = None
    mode: str | None = None
    threshold: float | None = None
    prompt_version: str | None = None
    max_pick: int = MAX_PICK
    client: Any = None

    def __post_init__(self) -> None:
        c = self.config
        self.mode = (self.mode or (c.decision_mode if c is not None else None) or "noul").strip().lower()
        if self.mode not in JEV_MODES:
            raise ValueError(f"unknown JEV_MODE {self.mode!r}; known: {', '.join(JEV_MODES)}")
        if self.threshold is None:
            self.threshold = c.threshold if c is not None and c.threshold is not None else JEV_THRESHOLD[self.mode]
        self.prompt_version = (self.prompt_version or (c.prompt_version if c is not None else None)
                               or JEV_PROMPT_VERSION)
        self.prompt = jev_prompt(self.mode, self.prompt_version)  # read now: a bad file fails at build, not per call

    def __call__(self, question: str, keys: list[str]) -> DecisionResult:
        if not keys:
            return DecisionResult(keys=[], calls=0)
        if self.client is None:
            if self.config is None:
                raise LLMKeyMissing("DecisionSelector needs client= or config= (an LLMConfig with api_key, base_url)")
            self.client = SystemOneClient(self.config)
        passages = [self.text_of[k] for k in keys]
        questions = (noul_questions(passages, self.prompt) if self.mode == "noul"
                     else choice_questions(passages, self.prompt))
        reply, calls = self.client.decide(self.prompt.state.replace("{question}", question), questions)
        scores = dict(zip(keys, probabilities(reply, self.mode, len(keys))))
        return DecisionResult(keys=cut(scores, self.threshold, self.max_pick), calls=calls,
                              usage=dict(reply.get("usage") or {}), reason=None, scores=scores)
