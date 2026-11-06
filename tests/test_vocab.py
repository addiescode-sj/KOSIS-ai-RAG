"""T2-30 (KOS-97): shared names for regions, breakdowns and units (no data files)."""

import pytest

from kosis_rag.parse import patterns
from kosis_rag.shared import filters, vocab
from kosis_rag.shared.vocab import LABEL_ALIASES, dimension_keys, unit_key

offline = pytest.mark.offline


@offline
def test_region_words_are_the_same_objects_as_before():
    assert vocab.SIDO is filters.SIDO and vocab.REGION_WORDS is patterns.REGION_WORDS
    assert vocab.SIDO_LABELS["강원도"] == "강원특별자치도"


@offline
@pytest.mark.parametrize("name, keys", [
    ("현거주지", ["현거주지"]), ("현 거주지", ["현거주지"]), ("현거주지별", ["현거주지"]),
    ("종사상의 지위", ["종사상지위"]), ("거처의종류", ["거처종류"]), ("직업(대분류)", ["직업"]),
    ("행정구역(시군구)", ["행정구역"]), ("학력", ["교육정도"]), ("장래 성", ["성"]),
    ("현 거주지 성", ["현거주지", "성"]),  # two breakdowns glued by a comma-split name
    ("특성", []),  # too vague to map: stays unmapped, never a refusal reason
])
def test_dimension_keys(name, keys):
    assert dimension_keys(name) == keys


@offline
def test_question_breakdowns_share_the_keys():
    for word, key in vocab.DIMENSION_WORDS.items():
        assert dimension_keys(key) == [key], word


@offline
@pytest.mark.parametrize("unit, key", [("명", "명"), (" 퍼센트", "%"), ("천명당", "천명당"), ("천명", "천명"), ("", None),
                                       ("15세 이상", None)])
def test_unit_key_only_respells(unit, key):
    assert unit_key(unit) == key


@offline
def test_one_person_household_is_a_household_size_label():
    assert LABEL_ALIASES["1인 가구"] == ("가구원수", "1인") and LABEL_ALIASES["7인 이상 가구"] == ("가구원수", "7인 이상")
