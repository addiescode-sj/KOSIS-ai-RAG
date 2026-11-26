"""T2-47: the stage 2 "together" verdict (prompt v4m): the 2-3 picked tables one question needs read together.

No API: the reply comes from Phase 1 MockClient. Checks the parser's rules, that v4 runs as before (no together, no
new metadata key), that v4m keeps v4's rules byte for byte, and that rerank_stage2 records stage2_together.
"""

import pytest
from langchain_core.documents import Document

from kosis_chat.llm.model import MockClient
from kosis_rag.rerank import selectors
from kosis_rag.rerank.hcx import HcxSelector, parse, parse_together, system_prompt
from kosis_rag.rerank.stage2 import rerank_stage2
from kosis_rag.schemas.resources import IndexSpec, SearchOptions

pytestmark = pytest.mark.offline

KEYS = ["a", "b", "c"]
TOGETHER = '{"picked": [2, 1, 3], "reason": null, "together": [2, 1]}'


def docs(keys=KEYS):
    return [Document(page_content=k, metadata={"key": k, "score": 0.9 - i / 10}, id=k) for i, k in enumerate(keys)]


def selector(reply, version="v4m"):
    return HcxSelector(text_of={k: f"표 {k}" for k in KEYS}, thinking="none", client=MockClient(reply),
                       prompt_version=version)


def test_together_is_kept_only_when_every_check_passes():
    assert parse_together(TOGETHER, [2, 1, 3]) == [2, 1]
    assert parse_together('{"picked": [3, 1], "together": ["1", "3"]}', [3, 1]) == [1, 3]
    for reply, picked in [
        ('{"picked": [3, 1], "together": null}', [3, 1]),
        ('{"picked": [3, 1], "together": [3]}', [3, 1]),  # one table is not a set
        ('{"picked": [3, 1], "together": [3, 3]}', [3, 1]),  # a repeat
        ('{"picked": [3, 1], "together": [3, 2]}', [3, 1]),  # outside picked: the whole list goes
        ('{"picked": [1, 2, 3, 4], "together": [1, 2, 3, 4]}', [1, 2, 3, 4]),  # phase 1 reads 3 tables at most
        ('{"picked": [1, 2], "together": [true, 2]}', [1, 2]),
        ('{"picked": [1, 2]}', [1, 2]),  # a v4 reply
        ("no json", [1, 2]),
    ]:
        assert parse_together(reply, picked) == [], reply


def test_v4_runs_as_before_and_v4m_names_the_tables():
    v4 = selector(TOGETHER, "v4")("질문", KEYS)
    assert v4.keys == ["b", "a", "c"] and v4.together == []  # v4 never reads together, even when a reply has it
    v4m = selector(TOGETHER)("질문", KEYS)
    assert v4m.keys == ["b", "a", "c"] and v4m.together == ["b", "a"]
    assert parse(TOGETHER, 3) == ([2, 1, 3], None)  # the picked parser is unchanged


def test_unknown_prompt_version_is_an_error():
    with pytest.raises(ValueError, match="STAGE2_PROMPT"):
        selector(TOGETHER, "v9")


def test_v4m_keeps_the_v4_rules_byte_for_byte():
    v4, v4m = system_prompt("stage2-hcx-v4.md"), system_prompt("stage2-hcx-v4m.md")
    assert v4m.startswith(v4.split("# 출력 형식")[0])  # sections 1-4 the same; section 5, output and examples differ
    assert "# 5. 함께 읽어야 하는 표 (together)" in v4m and '"together"' in v4m


def test_rerank_stage2_records_together_only_when_it_holds():
    out = rerank_stage2("질문", docs(), selector(TOGETHER), name="gpt")
    assert [d.metadata["key"] for d in out] == ["b", "a", "c"]
    assert all(d.metadata["stage2_together"] == ["b", "a"] for d in out)
    plain = rerank_stage2("질문", docs(), selector(TOGETHER, "v4"), name="gpt")
    assert all("stage2_together" not in d.metadata for d in plain)  # v4 metadata keeps its old keys


def test_the_option_reaches_the_numbered_selectors_and_the_label():
    for name in ("hcx", "gpt"):
        made = selectors.LLM_RERANKERS[name]({"a": "A"}, selectors.default_config(name), prompt_version="v4m")
        assert made.prompt_version == "v4m"
    assert selectors.LLM_RERANKERS["hcx"]({"a": "A"}, selectors.default_config("hcx")).prompt_version == "v4"
    spec = IndexSpec()
    assert SearchOptions(stage2="gpt").label(spec).endswith("gpt")
    assert SearchOptions(stage2="gpt", stage2_prompt="v4m").label(spec).endswith("gpt(v4m)")
