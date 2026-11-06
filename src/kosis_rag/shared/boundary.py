"""Base of the node-boundary models (T2-41; same settings as the Phase 1 boundary base of T1-28 S1b).

The stage schemas (parse/ retrieve/ rerank/ schemas.py, from S2) build their In/Out on this base:
- extra="forbid": an unknown keyword raises.
- strict: scalar fields (str, bool, int, Literal) are checked, never coerced (no 100 -> 100.0, no True -> 1).
- arbitrary types allowed, so a field may hold e.g. a LangChain Document (P2-D16 a).
- positional arguments fill the fields in declaration order.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class BoundaryModel(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, /, *args: Any, **data: Any) -> None:
        if args:
            names = list(type(self).model_fields)
            if len(args) > len(names):
                raise TypeError(f"{type(self).__name__}() takes {len(names)} positional arguments but {len(args)} were given")
            for name, value in zip(names, args):
                if name in data:
                    raise TypeError(f"{type(self).__name__}() got multiple values for argument {name!r}")
                data[name] = value
        super().__init__(**data)
