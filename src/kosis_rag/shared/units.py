# copied from Phase 1 src/kosis_ai/normalize.py@e248a45
"""The table-wide unit from the KOSIS table metadata (P2-D0): Phase 1 `_meta_unit` and `_known`, copied so
`metadata.py` (tables.jsonl) does not import Phase 1. tests/test_phase1_contract.py checks it gives the same
value as the Phase 1 function. Change it only together with Phase 1.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# Units seen in the provided data (docs/eda-full.md 2-4). Anything else in a unit column or in parentheses is not a
# unit: '월세없음', '15세 이상', '4년제이상', '미취학 포함' ...
KNOWN_UNITS = {
    "명", "건", "가구", "호", "개", "세", "년", "%", "천명", "원", "천원", "만원", "십만원", "백만원", "명/㎢", "㎢", "㎡",
    "배", "쌍", "점", "개사", "세대", "평", "척", "대", "여자백명당남자수",
}  # fmt: skip


def _known(unit: str | None) -> str | None:
    if not isinstance(unit, str):  # None or NaN from pandas
        return None
    text = unit.strip()
    return text if text in KNOWN_UNITS or (text.endswith("당") and len(text) <= 12) else None


@lru_cache(maxsize=256)
def meta_unit(table_id: str, data_dir: Path) -> str | None:
    path = data_dir / "table_metadata" / f"{table_id}.json"
    if not path.exists():
        return None
    return _known(json.loads(path.read_text(encoding="utf-8")).get("unitNmKor"))
