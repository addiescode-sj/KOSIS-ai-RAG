"""Names of the question (hq) collections, shared by the build that writes them and the search that opens them
(T2-41 S6a review: moved from retrieve/hq.py and retrieve/constants.py so the offline build needs no retrieve import).

In: build ids and embedder checkpoint ids. Out: collection names and the doc_format recorded on the collection.
Side effects: none.
"""

from __future__ import annotations

import hashlib
import re

from kosis_rag.shared.constants import COLLECTION_NAME

HQ_COLLECTION_PREFIX = "kosis_hq__"  # question collections: kosis_hq__<build_id>__<ckpt8>
HQ_DOC_FORMAT = "hq"  # collection metadata doc_format of a question index

_INVALID = re.compile(r"[^a-z0-9_-]+")


def ckpt8(ckpt_id: str | None) -> str:
    """8 characters naming the embedder weights: the sha part of model.ckpt_id ('<file>@<sha12>'), else a hash.

    Side effects: none. Node: none (names the collection).
    """
    if not ckpt_id:
        return "none"
    if "@" in ckpt_id and re.fullmatch(r"[0-9a-f]{8,}", ckpt_id.rsplit("@", 1)[1]):
        return ckpt_id.rsplit("@", 1)[1][:8]
    return "base" if ckpt_id == "base" else hashlib.sha256(ckpt_id.encode()).hexdigest()[:8]


def collection_name(build_id: str, ckpt_id: str | None) -> str:
    """kosis_hq__<build_id>__<ckpt8>, lower-cased with characters Chroma refuses replaced by '-' (3-512 chars,
    alphanumeric at both ends, no '..').
    Side effects: ValueError for a name Chroma refuses. Node: none (index CLI, load_resources).
    """
    name = _INVALID.sub("-", f"{HQ_COLLECTION_PREFIX}{build_id}__{ckpt8(ckpt_id)}".lower())[:512].strip("-_")
    if not COLLECTION_NAME.match(name) or ".." in name:
        raise ValueError(f"{name!r} is not a valid Chroma collection name")
    return name
