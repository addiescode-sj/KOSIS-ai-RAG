"""The fixed question-generation prompt (T2-39, plans/T2-39.md Idea 2): one table in, six questions out.

The same template is used for every table; only the table's n2 text and its field summary are filled in. No golden
question or answer is used anywhere in it. Changing the wording, the version or the generation settings changes
`prompt_hash()`, so replies cached under the old prompt are never reused (data/hq/cache/<prompt_hash8>/<model>/;
the table's own user message is keyed in the file name, hq.nodes.cache_path).

T2-41 S6a (from the old hq/prompt.py): the SYSTEM / USER / REASK texts are prompts/hq-v1.md (package data, read
through importlib.resources) — the bytes of the old string constants, so prompt_hash()[:8] stays "cb0b3c5b"
(tests/test_hq_prompt_bytes.py). The settings are in hq/constants.py.
"""

from __future__ import annotations

import functools
import hashlib
from importlib import resources
from typing import NamedTuple

from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared.fields import TableFields
from kosis_rag.shared.tabletext import CYCLE
from kosis_rag_build.hq.constants import MAX_LISTED_YEARS, MAX_TOKENS, PROMPT_FILE, PROMPT_VERSION, SEED, TEMPERATURE

USER_SEPARATOR = "\n<!-- user -->\n"  # between SYSTEM and USER in the prompt file (no text contains it)
REASK_SEPARATOR = "\n<!-- reask -->\n"  # between USER and REASK


class HqPrompt(NamedTuple):
    """The three texts of the prompt file: system message, user template ({doc}, {summary}), re-ask ({error})."""

    system: str
    user: str
    reask: str


@functools.cache
def prompt_texts(name: str = PROMPT_FILE) -> HqPrompt:
    """prompts/<name> split at its two separators (read once).

    Side effects: reads the package file; ValueError when the file does not hold each separator exactly once, in
    order. Node: prompt, validate (the re-ask).
    """
    text = (resources.files("kosis_rag_build.hq") / "prompts" / name).read_text(encoding="utf-8")
    system, sep, rest = text.partition(USER_SEPARATOR)
    user, sep2, reask = rest.partition(REASK_SEPARATOR)
    if not sep or not sep2 or USER_SEPARATOR in rest or REASK_SEPARATOR in reask or REASK_SEPARATOR in system:
        raise ValueError(f"{name}: expected {USER_SEPARATOR!r} and then {REASK_SEPARATOR!r}, once each")
    return HqPrompt(system, user, reask)


def prompt_hash() -> str:
    """sha256 of the fixed template, its version and the generation settings (not of any table's content).

    In: nothing. Out: 64 hex characters; its first 8 name the cache folder and go into the build id.
    Side effects: reads the prompt file once. Node: prompt.
    """
    p = prompt_texts()
    parts = [PROMPT_VERSION, p.system, p.user, p.reask, f"t={TEMPERATURE}", f"seed={SEED}", f"max={MAX_TOKENS}"]
    return hashlib.sha256("\n\x1e".join(parts).encode("utf-8")).hexdigest()


def years_text(years: list[int]) -> str:
    """'2015~2025 (매년)', '2000, 2005, 2010, 2015, 2020' or '1925~2020 중 45개 연도'; '알 수 없음' when empty.

    Side effects: none. Node: prompt.
    """
    if not years:
        return "알 수 없음"
    if len(years) == 1:
        return str(years[0])
    if years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}~{years[-1]} (매년)"
    if len(years) <= MAX_LISTED_YEARS:
        return ", ".join(map(str, years))
    return f"{years[0]}~{years[-1]} 중 {len(years)}개 연도"


def field_summary(record: TableRecord, fields: TableFields) -> str:
    """The facts the questions may use, one per line: measure, breakdowns, years, cycle, region level, 시도, unit.

    In: record = the table; fields = fields.table_fields(record). Out: Korean lines for the user message.
    Side effects: none. Node: prompt.
    """
    classes = fields.dimensions or [c.name for c in record.classes]
    cycle = "·".join(CYCLE.get(c, c) for c in fields.cycle) or "알 수 없음"
    if fields.regions is None:
        sido = "없음 (시도별 구분이 없는 표)"
    elif len(fields.regions) >= 17:
        sido = f"{len(fields.regions)}개 시도 전체"
    else:
        sido = ", ".join(fields.regions)
    return "\n".join([
        f"측정 대상: {fields.measure or record.info.table_name}",
        f"분류: {', '.join(classes) or '없음'}",
        f"수록 연도: {years_text(fields.available_years)}",
        f"주기: {cycle}",
        f"지역 수준: {'·'.join(['전국', *fields.region_level]) if fields.region_level else '전국 또는 구분 없음'}",
        f"시도: {sido}",
        f"단위: {fields.measure_unit or record.unit or '알 수 없음'}",
    ])


def build_messages(record: TableRecord, fields: TableFields, doc: str) -> list[dict[str, str]]:
    """[system, user] chat messages for one table.

    In: record, fields = the table and its fields; doc = its n2 embedding text. Out: two {"role", "content"} dicts.
    Side effects: reads the prompt file once. Node: prompt.
    """
    p = prompt_texts()
    return [{"role": "system", "content": p.system},
            {"role": "user", "content": p.user.format(doc=doc, summary=field_summary(record, fields))}]
