"""Query rewrite step functions (KOS-192): one LLM call that rewrites the question, and the code checks on it.

rewrite_question(RewriteInput, client) -> RewriteOutput calls the client once with the generate(system, user)
contract of llm/openai.py (JSON mode; rewrite_client builds it with temperature 0, MAX_TOKENS and a TIMEOUT) and never
raises: a failed call or an unreadable reply is status "error". check_rewrite(original, rewrite, original_parsed,
rewrite_parsed) is pure: the caller parses both questions with parse.node.parse_question and turns a returned reason
into status "rejected". The search graph does not call either yet (KOS-192 step 3); the offline scripts do.
"""

from __future__ import annotations

import functools
import json
import re
import time
import unicodedata
from importlib import resources
from typing import Any

from kosis_rag.llm.base import STATUS_OK
from kosis_rag.llm.config import LLMConfig
from kosis_rag.llm.providers import build_client
from kosis_rag.parse.schemas import ParsedQuery
from kosis_rag.rewrite.constants import (CLASS_NUMBER_PATTERNS, MAX_CHARS, MAX_TABLE_NAMES, MAX_TOKENS, PARTICLES,
                                         PROMPT_FILE, REPLY_STATUSES, TABLES_HEADER, TEMPERATURE, TIME_UNITS, TIMEOUT)
from kosis_rag.rewrite.schemas import RewriteInput, RewriteOutput

DIGITS = re.compile(r"\d+")  # a number in a question: one run of digits
WORD = re.compile(r"\w+")  # a word: letters and digits (Hangul counts as letters); whitespace and punctuation split
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # 1,000 -> 1000
QUARTER = re.compile(r"(\d)\s*/\s*4\s*분기")  # 2/4분기 -> 2분기
BARE_YEAR = re.compile(r"(?:19|20)\d\d")  # a whole digit run read as a year
TIME_UNIT = re.compile("|".join(TIME_UNITS))  # matched right after a number
CLASS_NUMBER = re.compile(r"(?<!\d)(?:" + "|".join(CLASS_NUMBER_PATTERNS) + ")")


@functools.cache
def system_prompt(name: str = PROMPT_FILE) -> str:
    """The rewriter's system prompt: rewrite/prompts/<name> as UTF-8 text (read once).

    Side effects: reads the package data file (FileNotFoundError when it is missing). Node: none yet (KOS-192).
    """
    return (resources.files("kosis_rag.rewrite") / "prompts" / name).read_text(encoding="utf-8")


def user_message(question: str, table_names: list[str] | None = None) -> str:
    """The user message: `질문: {question}`; with table names (R2) also the block `참고 후보 통계표:` and one
    `- name` per line, the first MAX_TABLE_NAMES in the order given. Side effects: none. Node: none yet (KOS-192)."""
    lines = [f"질문: {question}"]
    if table_names:
        lines += [TABLES_HEADER, *(f"- {name}" for name in table_names[:MAX_TABLE_NAMES])]
    return "\n".join(lines)


def parse_reply(text: str) -> tuple[str | None, str]:
    """Reply -> (rewrite, "ok") or (None, "none"), read from the first JSON object in it (as rerank.hcx.parse); the
    keys may come in any order.

    A reply without "status" but with a non-empty "rewrite" counts as "ok". The rewrite is stripped ("" stays "", and
    check_rewrite rejects it as empty). Raises ValueError when no JSON object is found, the object has an unknown
    status or the rewrite is not text. Side effects: none. Node: none yet (KOS-192).
    """
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:80]!r}")
    data, _ = json.JSONDecoder().raw_decode(text, start)  # the first object; a model sometimes repeats it
    if not isinstance(data, dict):
        raise ValueError(f"reply is not a JSON object: {text[:80]!r}")
    rewrite, status = data.get("rewrite"), data.get("status")
    if rewrite is not None and not isinstance(rewrite, str):
        raise ValueError(f"rewrite is not text: {rewrite!r}")
    if status is None and rewrite and rewrite.strip():
        status = "ok"
    if status not in REPLY_STATUSES:
        raise ValueError(f"unknown status {status!r}; known: {', '.join(REPLY_STATUSES)}")
    return (None, "none") if status == "none" else ((rewrite or "").strip(), "ok")


def _tokens(usage: dict[str, Any], *names: str) -> int | None:
    """The first of the usage fields that is a whole number (OpenAI input_tokens, HCX promptTokens ...), or None."""
    value = next((usage[n] for n in names if usage.get(n) is not None), None)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def rewrite_question(inp: RewriteInput, client: Any, *, prompt: str = PROMPT_FILE) -> RewriteOutput:
    """Rewrite the question in official statistics terms with one LLM call.

    In: inp.question; inp.table_names = names of the top stage 1 candidates (R2; at most MAX_TABLE_NAMES are sent) or
        [] (R1); client = an object with generate(system, user) -> a result with .text (and .status / .usage / .error
        when it has them): a client from rewrite_client in production, a fake in tests; prompt = the system prompt
        file under rewrite/prompts (default PROMPT_FILE = rewrite-v1.md; KOS-192 experiments pass rewrite-v2.md).
    Out: RewriteOutput. status "used" with the rewrite when the model gave one ("rewrite" may still fail
        check_rewrite: the caller marks it "rejected"), "none" (rewrite None) when the model said there is nothing to
        rewrite, "error" (rewrite None, reason = the error text) when the call raised, returned a status other than
        ok, or its reply could not be read. seconds = the call and the parse; tokens from the client's usage when
        given (kept on an unreadable reply).
    Side effects: one LLM call through `client`; never raises.
    Node: none yet (KOS-192 step 3 adds the rewrite node).
    """
    started = time.perf_counter()
    usage: dict[str, Any] = {}
    try:
        result = client.generate(system_prompt(prompt), user_message(inp.question, inp.table_names))
        usage = dict(getattr(result, "usage", None) or {})
        if getattr(result, "status", STATUS_OK) != STATUS_OK:
            raise RuntimeError(f"call failed ({result.status}): {getattr(result, 'error', None)}")
        rewrite, status = parse_reply(result.text or "")
        out_status, reason = ("used" if status == "ok" else "none"), None
    except Exception as exc:  # a failed rewrite falls back to the original question; the search goes on
        rewrite, out_status, reason = None, "error", f"{type(exc).__name__}: {exc}"
    return RewriteOutput(rewrite=rewrite, status=out_status, reason=reason, seconds=time.perf_counter() - started,
                         input_tokens=_tokens(usage, "input_tokens", "promptTokens"),
                         output_tokens=_tokens(usage, "output_tokens", "completionTokens"))


def _prepare(text: str) -> str:
    """NFKC, no thousands separators, "N/4분기" read as "N분기": the form numbers are compared in."""
    return QUARTER.sub(r"\1분기", THOUSANDS.sub("", unicodedata.normalize("NFKC", text)))


def _strip_particle(word: str) -> str:
    """The word without one particle at its end (PARTICLES, longest first); a word that is only the particle stays."""
    for p in PARTICLES:
        if word.endswith(p) and len(word) > len(p):
            return word[: -len(p)]
    return word


def normalize(text: str) -> str:
    """text as check_rewrite compares it for "unchanged": NFKC, lower-cased, whitespace and punctuation dropped, one
    particle stripped from the end of each word (은 는 이 가 을 를 의 에 에서 으로 로 와 과 도)."""
    return "".join(_strip_particle(w) for w in WORD.findall(unicodedata.normalize("NFKC", text).lower()))


def numbers(text: str) -> list[tuple[str, bool, tuple[int, int]]]:
    """Every number in an already prepared text as (value without leading zeros, is a time number, span).

    A time number is a bare 19xx / 20xx digit run, or a number followed right away by 년 | 년대 | 월 | 분기 | 반기 |
    개월 | 일 | 주 (TIME_UNITS). Side effects: none.
    """
    return [(str(int(m[0])), bool(BARE_YEAR.fullmatch(m[0]) or TIME_UNIT.match(text, m.end())), m.span())
            for m in DIGITS.finditer(text)]


def check_rewrite(original: str, rewrite: str | None, original_parsed: ParsedQuery,
                  rewrite_parsed: ParsedQuery) -> str | None:
    """Why the rewrite must be thrown away, or None when it may be used (plan §4.4, review of 11/13).

    In: original = the user question; rewrite = the model's rewrite; original_parsed / rewrite_parsed = the two
        questions parsed by parse.node.parse_question (the caller parses them, so this stays pure).
    Out: the first reason that applies, in this order (constants.REJECT_REASONS):
        "empty" (nothing but whitespace) · "too_long" (more than MAX_CHARS characters after stripping) ·
        "unchanged" (equal to the original under normalize: whitespace, punctuation and word-final particles dropped) ·
        numbers, compared after _prepare (NFKC, no thousands separators, N/4분기 = N분기):
        "new_number" (a time number of the rewrite that the original does not have, or any other number new in the
        rewrite that is not inside a classification term: 1인가구, 2인 이상 가구, 65세 이상, 15~64세, 5세별, 20대) ·
        "dropped_number" (a time number of the original missing from the rewrite: "21년" must stay "21년") ·
        "condition_years" / "condition_period" (period_start or period_end) (years compared as sets) ·
        "condition_cycle" (the original has a cycle and the rewrite's is another or none; a cycle the rewrite adds is
        allowed) · "condition_recent" (ParsedQuery.recent differs) · "condition_regions" (as sets: missing and added
        both count). Added dimensions and measures are allowed. None = keep it.
    Side effects: none. Node: none yet (KOS-192).
    """
    text = (rewrite or "").strip()
    if not text:
        return "empty"
    if len(text) > MAX_CHARS:
        return "too_long"
    if normalize(text) == normalize(original):
        return "unchanged"
    before, after = _prepare(original), _prepare(text)
    a_nums, b_nums = numbers(before), numbers(after)
    a_time = {v for v, is_time, _ in a_nums if is_time}
    b_time = {v for v, is_time, _ in b_nums if is_time}
    if b_time - a_time:
        return "new_number"
    if a_time - b_time:
        return "dropped_number"
    known = {v for v, _, _ in a_nums}
    terms = [m.span() for m in CLASS_NUMBER.finditer(after)]
    for value, is_time, (start, end) in b_nums:
        if not is_time and value not in known and not any(s <= start and end <= e for s, e in terms):
            return "new_number"
    a, b = original_parsed, rewrite_parsed
    if set(a.years) != set(b.years):
        return "condition_years"
    if (a.period_start, a.period_end) != (b.period_start, b.period_end):
        return "condition_period"
    if a.cycle is not None and b.cycle != a.cycle:
        return "condition_cycle"
    if a.recent != b.recent:
        return "condition_recent"
    if set(a.regions) != set(b.regions):
        return "condition_regions"
    return None


def rewrite_client(config: LLMConfig) -> Any:
    """The rewriter's LLM client: the config's provider (llm.providers.build_client) with temperature TEMPERATURE and
    max_tokens MAX_TOKENS whatever the config says, and timeout TIMEOUT when the config sets none (the stage 2 config
    is reused as is otherwise: model, base_url, key).

    In: config = an LLMConfig (the scripts pass the stage 2 "gpt" config: the local OpenAI-compatible Gemma server).
    Out: a client with generate(system, user). Side effects: none until called (HCXClient checks its key when made).
    Node: none yet (KOS-192).
    """
    update: dict[str, Any] = {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    if config.timeout is None:
        update["timeout"] = TIMEOUT
    return build_client(config.model_copy(update=update))
