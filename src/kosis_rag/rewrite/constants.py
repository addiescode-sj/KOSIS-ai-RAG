"""Constants of the query rewrite step (KOS-192)."""

from __future__ import annotations

PROMPT_VERSION = "v1"
PROMPT_FILE = f"rewrite-{PROMPT_VERSION}.md"  # rewrite/prompts/<file>: the rewriter's system prompt
# KOS-192 11/17: v2 keeps every condition (age, sex, 산업 vs 직업), uses R2 candidates for words only (no copied table
# names), adds verbatim region names and "none" for proxy measures; experiments choose it with prompt=PROMPT_FILE_V2
PROMPT_FILE_V2 = "rewrite-v2.md"
TEMPERATURE = 0.0  # one deterministic rewrite per question
MAX_TOKENS = 256  # the reply is one JSON line with one short phrase; room for a model that adds a little
TIMEOUT = 30.0  # seconds per call when the LLM config sets none (rewrite_client)
MAX_CHARS = 120  # a longer rewrite is thrown away (check_rewrite); the prompt asks for 80 or fewer
MAX_TABLE_NAMES = 10  # R2: at most this many stage 1 candidate names reach the rewriter (user_message cuts the list)
TABLES_HEADER = "참고 후보 통계표:"  # the user message block of R2 (the prompt's input format names it)
REPLY_STATUSES = ("ok", "none")  # what the model may answer in "status"
# check_rewrite's reasons, in the order it checks them (the first that applies is returned)
REJECT_REASONS = ("empty", "too_long", "unchanged", "new_number", "dropped_number", "condition_years",
                  "condition_period", "condition_cycle", "condition_recent", "condition_regions")
# particles stripped from the end of each word before the "unchanged" comparison (longest first)
PARTICLES = ("에서", "으로", "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도")
# a number followed by one of these is a time number (so is a bare 19xx / 20xx); longest first
TIME_UNITS = ("년대", "개월", "분기", "반기", "년", "월", "일", "주")
# classification terms whose number may appear in a rewrite without being in the question (1인가구, 65세 이상, 20대)
CLASS_NUMBER_PATTERNS = (r"[1-9]인 ?(?:이상 ?)?가구", r"\d{1,3}~\d{1,3}세", r"\d{1,2}세(?:별| ?계급)",
                         r"\d{1,3}세(?: ?(?:이상|이하|미만))?", r"\d{2}대")
