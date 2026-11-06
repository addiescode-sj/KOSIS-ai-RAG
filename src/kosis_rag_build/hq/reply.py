"""Reading the question LLM's reply (T2-41 S6b: from hq/nodes.py, so hq/report.py imports it without a cycle).

A leaf: imports only the standard library. Used by hq/nodes.py (validate) and hq/report.py (the reply samples).
"""

from __future__ import annotations

import json


def parse_reply(raw: str) -> dict:
    """The first JSON object in a reply.

    In: raw = the reply text. Out: the decoded object. Side effects: raises ValueError when there is no JSON object.
    Node: validate.
    """
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {raw[:80]!r}")
    try:
        data, _ = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"broken JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data
