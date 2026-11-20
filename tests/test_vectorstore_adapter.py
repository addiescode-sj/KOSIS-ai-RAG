"""T2-44: the vector store adapter (PostgreSQL + pgvector; was the Chroma client, T2-23 / T2-41 S1).

The offline tests check the address handling, the one filter the search sends and the SQL it becomes; no database is
contacted (verify.sh stays offline). The `pgvector` tests run against a real database only when TEST_DATABASE_URL is
set (e.g. `docker compose up -d postgres` and TEST_DATABASE_URL=postgresql://kosis:kosis@localhost:5432/kosis); each
uses its own collection and deletes it.
"""

import os
import uuid

import pytest

from kosis_rag.adapters import vectorstore
from kosis_rag.schemas.resources import DEFAULT_DATABASE_URL, VectorStoreConfig

offline = pytest.mark.offline


@offline
def test_config_defaults_to_the_compose_postgres_on_the_host():
    config = VectorStoreConfig()
    assert (config.url, config.collection) == (DEFAULT_DATABASE_URL, None)
    assert "localhost:5432" in config.url


@offline
def test_sqlalchemy_urls_are_accepted_and_passwords_are_not_printed():
    url = "postgresql+psycopg://kosis:secret@postgres:5432/kosis"
    assert vectorstore.plain_url(url) == "postgresql://kosis:secret@postgres:5432/kosis"
    assert vectorstore.safe_url(url) == "postgresql://kosis:***@postgres:5432/kosis"
    assert vectorstore.safe_url("postgresql://localhost/kosis") == "postgresql://localhost/kosis"


@offline
def test_only_the_key_in_filter_is_taken():
    assert vectorstore.key_filter(None) is None and vectorstore.key_filter({}) is None
    assert vectorstore.key_filter({"key": {"$in": ["a", "b"]}}) == ["a", "b"]
    for bad in ({"key": "a"}, {"key": {"$nin": ["a"]}}, {"$and": [{"key": {"$in": ["a"]}}]}, {"type": {"$in": ["x"]}}):
        with pytest.raises(ValueError, match="unsupported filter"):
            vectorstore.key_filter(bad)


@offline
def test_query_sql_filters_by_key_and_breaks_ties_by_id():
    sql, tail = vectorstore.query_sql(5, None)
    assert "key = ANY" not in sql and "ORDER BY distance, id LIMIT" in sql and tail == [5]
    sql, tail = vectorstore.query_sql(3, ["k1", "k2"])
    assert "collection = %s AND key = ANY(%s)" in sql and tail == [["k1", "k2"], 3]
    assert "<=>" in sql  # cosine distance, Chroma's cosine space


@offline
def test_client_says_how_to_start_the_server_when_it_is_down(monkeypatch):
    psycopg = pytest.importorskip("psycopg")

    def refuse(*_, **__):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(psycopg, "connect", refuse)
    with pytest.raises(ConnectionError, match=r"kosis:\*\*\*@localhost:5432.*docker compose up -d postgres"):
        vectorstore.vector_client(VectorStoreConfig())


@offline
def test_adapter_reads_no_environment(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    monkeypatch.setenv("DATABASE_URL", "postgresql://elsewhere/x")
    seen = []

    def refuse(url, **_):
        seen.append(url)
        raise psycopg.OperationalError("no")

    monkeypatch.setattr(psycopg, "connect", refuse)
    with pytest.raises(ConnectionError):
        vectorstore.vector_client(VectorStoreConfig())
    assert seen == [DEFAULT_DATABASE_URL]


# --- against a real database (TEST_DATABASE_URL) ---

TEST_URL = os.environ.get("TEST_DATABASE_URL")
pgvector = pytest.mark.skipif(not TEST_URL, reason="set TEST_DATABASE_URL to run the PostgreSQL + pgvector tests")


@pytest.fixture
def store():
    client = vectorstore.vector_client(VectorStoreConfig(url=TEST_URL))
    name = f"test_{uuid.uuid4().hex[:12]}"
    yield client, name
    client.delete_collection(name)


def _unit(*xs):
    n = sum(x * x for x in xs) ** 0.5
    return [x / n for x in xs]


ROWS = {  # id -> (key, vector); the query is (1, 0, 0)
    "t1": ("A", _unit(1, 0, 0)), "t2": ("B", _unit(1, 1, 0)), "t3": ("C", _unit(0, 1, 0)), "t4": ("A", _unit(0, 0, 1)),
}


def _load(client, name):
    c = client.get_or_create_collection(name, metadata={"embed_model": "fake", "dim": 3, "hnsw:space": "cosine"})
    ids = list(ROWS)
    c.upsert(ids=ids, documents=[f"doc {i}" for i in ids], metadatas=[{"key": ROWS[i][0], "n": 1} for i in ids],
             embeddings=[ROWS[i][1] for i in ids])
    return c


@pgvector
def test_query_returns_chromas_shape_closest_first(store):
    client, name = store
    c = _load(client, name)
    res = c.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=3, include=["documents", "metadatas", "distances"])
    assert res["ids"] == [["t1", "t2", "t3"]]
    assert res["documents"][0][0] == "doc t1" and res["metadatas"][0][0] == {"key": "A", "n": 1}
    d = res["distances"][0]
    assert d[0] == pytest.approx(0.0, abs=1e-6) and d[1] == pytest.approx(1 - 2 ** -0.5, abs=1e-6)
    assert d[2] == pytest.approx(1.0, abs=1e-6)


@pgvector
def test_key_filter_gives_every_matching_row_up_to_k(store):
    """KOS-138: a filtered search must not come back short (an exact scan has no HNSW candidate cut)."""
    client, name = store
    c = _load(client, name)
    res = c.query(query_embeddings=[[0.0, 1.0, 0.0]], n_results=2, where={"key": {"$in": ["A"]}})
    assert res["ids"] == [["t1", "t4"]]  # both A rows, though t2 and t3 are closer to the query
    res = c.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=10, where={"key": {"$in": ["A", "C"]}})
    assert res["ids"][0] == ["t1", "t3", "t4"]


@pgvector
def test_build_side_calls_behave_like_chroma(store):
    client, name = store
    c = _load(client, name)
    assert c.count() == 4 and client.get_collection(name).metadata["dim"] == 3
    assert name in [x.name for x in client.list_collections()]
    assert c.get(include=[])["ids"] == ["t1", "t2", "t3", "t4"]
    got = c.get(ids=["t2"], include=["embeddings", "documents"])
    assert got["documents"] == ["doc t2"] and list(got["embeddings"][0]) == pytest.approx(ROWS["t2"][1], abs=1e-7)
    c.upsert(ids=["t2"], documents=["new"], metadatas=[{"key": "B"}], embeddings=[ROWS["t2"][1]])  # overwrite
    assert c.count() == 4 and c.get(ids=["t2"])["documents"] == ["new"]
    c.delete(ids=["t3", "t4"])
    assert c.count() == 2
    again = client.get_or_create_collection(name, metadata={"other": 1})  # an existing collection keeps its record
    assert again.metadata["embed_model"] == "fake"
    again.modify(metadata={"embed_model": "fake", "dim": 3, "index_version": "x"})
    assert client.get_collection(name).metadata["index_version"] == "x"
    with pytest.raises(ValueError, match="3-dimension"):
        c.upsert(ids=["bad"], documents=["d"], metadatas=[{"key": "Z"}], embeddings=[[1.0, 0.0]])
    with pytest.raises(vectorstore.CollectionNotFound):
        client.get_collection(name + "_missing")


@pgvector
def test_the_build_load_function_runs_on_the_store(store):
    """kosis_rag_build.vector_index.load (tables, chunks and hq-index all load through it) is unchanged."""
    from kosis_rag_build.vector_index import load

    client, name = store
    ids = list(ROWS)
    record = {"embed_model": "fake", "dim": 3, "hnsw:space": "cosine", "n_docs": 4}
    args = ([f"doc {i}" for i in ids], [{"key": ROWS[i][0]} for i in ids], [ROWS[i][1] for i in ids])
    assert load(client, ids, *args, name, record) == 4
    assert load(client, ids[:2], *(a[:2] for a in args), name, record) == 2  # stale ids are dropped on reload
    assert client.get_collection(name).get(include=[])["ids"] == ["t1", "t2"]
