"""T2-41 parity gate (2): the mock cases give the frozen fixtures byte for byte (tests/fixtures/parity/README.md).

PARITY_UPDATE=1 writes the fixtures instead of comparing (S0 only; a later step that needs it has changed behavior).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from tests.parity_cases import CASE_NAMES, CASES, KINDS, case_texts  # noqa: E402

pytestmark = pytest.mark.offline

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "parity"
UPDATE = os.environ.get("PARITY_UPDATE", "").strip() == "1"


def _path(name: str, kind: str) -> Path:
    return FIXTURES / f"{name}.{kind}.json"


@pytest.mark.parametrize("case", CASES, ids=CASE_NAMES)
def test_case_matches_its_fixtures(case):
    texts = case_texts(case)
    for kind in KINDS:
        path = _path(case.name, kind)
        data = texts[kind].encode("utf-8")
        if UPDATE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            continue
        assert path.is_file(), f"missing fixture {path.name}: run once with PARITY_UPDATE=1 (S0 only)"
        assert path.read_bytes() == data, f"{path.name} differs from the frozen fixture"


def test_a_case_gives_the_same_bytes_twice():
    for case in CASES:
        assert case_texts(case) == case_texts(case), case.name


def test_no_stray_fixtures():
    expected = {_path(n, k).name for n in CASE_NAMES for k in KINDS}
    found = {p.name for p in FIXTURES.glob("*.json")}
    assert found == expected, f"unexpected: {sorted(found - expected)}, missing: {sorted(expected - found)}"
