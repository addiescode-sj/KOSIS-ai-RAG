"""Fixed settings of the question build (T2-39; T2-41 S6a: from hq/prompt.py, hq/rules.py and hq/graph.py).

Generation settings and the prompt version are part of prompt.prompt_hash(): changing one changes the cache folder
and the build id. Rule thresholds and the filter defaults are fixed, never tuned on golden questions.
"""

from __future__ import annotations

# prompt (hq/prompt.py; part of prompt_hash)
PROMPT_VERSION = "hq-v1"
PROMPT_FILE = "hq-v1.md"  # prompts/<file>: SYSTEM, USER and REASK of PROMPT_VERSION
TEMPERATURE = 0.7  # some variety between the questions of one table; the seed keeps a rerun close
SEED = 0
MAX_TOKENS = 1024
COUNTS = {"concept": 2, "condition": 2, "distinct": 1, "keyword": 1}  # D5: six questions (written out in the prompt)
MAX_LISTED_YEARS = 40  # longer gappy year lists are summarized as a range with a count

# rules (hq/rules.py)
NAME_COPY = 0.9
DUPLICATE = 0.95

# build (hq/nodes.py, graph/hq_build.py)
FILTER_FRAC = 0.3  # Doc2Query−− default share of the lowest-scored questions dropped
MIN_KEEP = 2  # questions every table keeps (when it has that many after the rules)
CONCURRENCY = 8
MAX_ATTEMPTS = 2  # the first reply and one re-ask
EXAMPLES = 20
