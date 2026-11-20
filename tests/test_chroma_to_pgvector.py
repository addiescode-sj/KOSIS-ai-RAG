"""T2-44: scripts/chroma_to_pgvector.py copies a collection without re-embedding and checks it.

The source is an in-memory stand-in for a Chroma collection (paged get, exact cosine query); the destination is the
real pgvector store, so these run only when TEST_DATABASE_URL is set (see tests/test_vectorstore_adapter.py).
"""

import importlib.util
import math
import os
import random
import uuid
from pathlib import Path

import pytest

from kosis_rag.adapters import vectorstore
from kosis_rag.schemas.resources import VectorStoreConfig

ROOT = Path(__file__).resolve().parents[1]
TEST_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="set TEST_DATABASE_URL to run the PostgreSQL + pgvector tests")


def _tool():
    spec = importlib.util.spec_from_file_location("chroma_to_pgvector", ROOT / "scripts" / "chroma_to_pgvector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def _cos_dist(a, b):
    return 1 - sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class SourceCollection:
    """What the script reads from a Chroma collection."""

    def __init__(self, name, rows, metadata):
        self.name, self.rows, self.metadata = name, rows, metadata  # rows: id -> (doc, meta, vec)

    def get(self, include, limit, offset):
        ids = sorted(self.rows)[offset:offset + limit]
        return {"ids": ids, "documents": [self.rows[i][0] for i in ids], "metadatas": [self.rows[i][1] for i in ids],
                "embeddings": [self.rows[i][2] for i in ids]}

    def query(self, query_embeddings, n_results, include):
        q = query_embeddings[0]
        ranked = sorted(self.rows, key=lambda i: (_cos_dist(self.rows[i][2], q), i))[:n_results]
        return {"ids": [ranked]}


class Source:
    def __init__(self, collections):
        self.collections = {c.name: c for c in collections}

    def get_collection(self, name):
        return self.collections[name]


def _rows(n, dim, seed):
    rng = random.Random(seed)
    return {f"101_DT_{i:04d}": (f"표 {i}", {"key": f"101_DT_{i:04d}", "f_years": [2020, 2021], "f_national": i % 2 == 0},
                                [rng.uniform(-1, 1) for _ in range(dim)]) for i in range(n)}


@pytest.fixture
def dst():
    client = vectorstore.vector_client(VectorStoreConfig(url=TEST_URL))
    names = []
    yield client, names
    for name in names:
        client.delete_collection(name)


def test_collections_are_copied_as_they_are_and_checked(dst):
    client, names = dst
    name = f"test_mig_{uuid.uuid4().hex[:8]}"
    names.append(name)
    meta = {"embed_model": "fake", "doc_format": "x", "dim": 8, "hnsw:space": "cosine", "n_docs": 1203}
    src = Source([SourceCollection(name, _rows(1203, 8, 1), meta)])  # more than two pages
    lines = []
    assert TOOL.migrate(src, client, [name], log=lines.append) == 0, lines
    assert lines[0].startswith(f"{name}: ok - 1203 rows, dim 8") and not any("note" in x for x in lines), lines
    c = client.get_collection(name)
    assert c.count() == 1203 and c.metadata == meta
    got = c.get(ids=["101_DT_0007"], include=["metadatas", "embeddings"])
    assert got["metadatas"] == [src.collections[name].rows["101_DT_0007"][1]]
    # a rerun after rows were removed in Chroma gives Chroma's collection again
    del src.collections[name].rows["101_DT_0000"]
    assert TOOL.migrate(src, client, [name], log=lines.append) == 0
    assert c.count() == 1202


def test_a_vector_length_that_disagrees_with_the_record_is_refused(dst):
    client, names = dst
    name = f"test_mig_{uuid.uuid4().hex[:8]}"
    src = Source([SourceCollection(name, _rows(3, 4, 2), {"dim": 1024})])
    lines = []
    assert TOOL.migrate(src, client, [name], log=lines.append) == 1
    assert "FAILED" in lines[0] and "1024" in lines[0]
    with pytest.raises(vectorstore.CollectionNotFound):
        client.get_collection(name)
