"""Constants of the rule parser (T2-29 C7; T2-41 S2: from query.py)."""

from __future__ import annotations

# C7, two layers: only these concepts filter (words one of which the table's counted thing must hold); every other
# concept goes to TableFilter.soft_measures and only ranks. 혼인상태·출생지 are breakdowns, not these.
HARD_MEASURE_WORDS = {"혼인": ["혼인", "초혼", "재혼"], "이혼": ["이혼"], "출생": ["출생", "출산", "생아"], "사망": ["사망"]}
SOFT_MEASURE_WORDS = {"취업자": ["취업"], "고령자": ["고령", "노인"], "이동자": ["이동"], "부인": ["부인", "기혼여성"],
                      "기혼여성": ["기혼여성", "부인"]}  # others (인구, 가구, 외국인, 주택, 1인가구) are their own word
MAX_RESULTS = 10  # candidates handed on after filtering
