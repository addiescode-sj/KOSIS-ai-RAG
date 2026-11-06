"""The guide prompt (T2-41 S5: chain.PROMPT): prompts/guide-v1.md, read through importlib.resources.

The file holds the system message, a separator line and the human message, each the bytes of the old PROMPT
strings (tests/test_prompts_bytes.py). The human message is a template over {context} and {question}.
"""

from __future__ import annotations

import functools
from importlib import resources

from langchain_core.prompts import ChatPromptTemplate

PROMPT_FILE = "guide-v1.md"
SEPARATOR = "\n<!-- human -->\n"  # between the system and the human message (neither contains it)


@functools.cache
def guide_messages(name: str = PROMPT_FILE) -> tuple[str, str]:
    """(system, human) of prompts/<name> as UTF-8 text (read once). Side effects: reads the file; ValueError when the
    file does not hold exactly one separator."""
    text = (resources.files("backend.app.respond") / "prompts" / name).read_text(encoding="utf-8")
    parts = text.split(SEPARATOR)
    if len(parts) != 2:
        raise ValueError(f"{name}: expected one {SEPARATOR!r} line between the system and the human message")
    return parts[0], parts[1]


@functools.cache
def guide_prompt(name: str = PROMPT_FILE) -> ChatPromptTemplate:
    """The guide prompt: system + human ({context}, {question}), as the old chain.PROMPT. Node: guide."""
    system, human = guide_messages(name)
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])
