"""KOS-192: SearchOptions.cond_scope keeps only the years and the period range of the question's filter."""

import pytest

from kosis_rag.parse.node import scoped
from kosis_rag.schemas.resources import IndexSpec, SearchOptions
from kosis_rag.shared.filters import TableFilter

FULL = TableFilter(years=[2015], mode="strict", cycle="Y", regions=["부산광역시"], dimensions=["성"],
                   measures=["이혼"], soft_measures=["인구"])


def test_all_keeps_the_filter():
    assert scoped(FULL, "all") == FULL


def test_years_keeps_only_years_and_period():
    f = scoped(FULL, "years")
    assert f == TableFilter(years=[2015], mode="strict")
    assert not f.is_soft()
    assert FULL.regions == ["부산광역시"]  # the input is left as it is


def test_years_keeps_a_period_range():
    f = scoped(TableFilter(period_from=2015, period_to=2020, mode="partial", measures=["혼인"]), "years")
    assert (f.period_from, f.period_to, f.mode, f.measures) == (2015, 2020, "partial", [])


def test_years_without_years_is_empty():
    assert scoped(TableFilter(regions=["서울특별시"], measures=["출생"]), "years").is_empty()


def test_unknown_scope():
    with pytest.raises(ValueError):
        scoped(FULL, "regions")


def test_default_and_label():
    assert SearchOptions().cond_scope == "all"
    spec = IndexSpec()
    assert SearchOptions().label(spec) == "hybrid+cond"
    assert SearchOptions(cond_scope="years").label(spec) == "hybrid+cond(years)"
