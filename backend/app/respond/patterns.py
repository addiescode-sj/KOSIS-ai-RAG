"""The regexes the guide check reads (T2-41 S5: chain._TBL_ID _YEAR _REF_RANGE _REF, public names now)."""

from __future__ import annotations

import re

TBL_ID = re.compile(r"(?<![0-9A-Za-z_])DT_[0-9A-Za-z_]+")  # not \b: Korean letters count as \w ("표DT_X", T2-38)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# "[1]", "[1, 2]", "[1~3]", full-width "［1］"; "[1]~[3]" is read as a range too (T2-39)
REF_RANGE = re.compile(r"[\[［]\s*(\d{1,3})\s*[\]］]\s*[~\-]\s*[\[［]\s*(\d{1,3})\s*[\]］]")
REF = re.compile(r"[\[［]\s*(\d{1,3}(?:\s*[,，~\-]\s*\d{1,3})*)\s*[\]］]")
