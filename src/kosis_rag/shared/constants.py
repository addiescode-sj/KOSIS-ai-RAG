"""Constants shared by the search stages (T2-41). S2: COLLECTION_NAME, RRF_C, DESC_SUFFIX; S3: RRF_C is rerank's too; S4 adds OFF ..."""

from __future__ import annotations

import re

# Chroma collection names: 3-512 chars, alphanumeric at both ends (from index.py; the table and question collections)
COLLECTION_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,510}[a-z0-9]$")
RRF_C = 60  # reciprocal rank fusion constant: RRF = sum of w / (RRF_C + rank) (retrieve.fuse, rerank.stage1 blend)
DESC_SUFFIX = "-desc"  # question index build id suffix of the description-only index (S1-desc; from hq/index.py)
