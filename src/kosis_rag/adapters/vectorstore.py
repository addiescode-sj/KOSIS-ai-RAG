"""The one way stage 2 code reaches the vector store: PostgreSQL + pgvector (T2-44; was the ChromaDB HTTP client).

The client and its collections keep the small part of the Chroma client API the search and the builds use, so those
callers did not change:

    client.get_collection(name) / get_or_create_collection(name, metadata=...) / list_collections() /
    get_max_batch_size() / delete_collection(name)
    collection.name / .metadata / count() / modify(metadata=...) /
    query(query_embeddings=[v], n_results=k, include=[...], where={"key": {"$in": [...]}}) /
    get(ids=None, include=[...], limit=None, offset=None) / upsert(ids, documents, metadatas, embeddings) / delete(ids)

Rows live in two tables (DDL below, the same as the backend's alembic migration 0002_pgvector):
vector_collection(name, metadata) and vector_item(collection, id, key, document, metadata, embedding). `key` is the
table key of every row (tables, region chunks, hypothetical questions), so the one filter the search sends — the
passing table keys — is a plain `key = ANY(...)`. There is no approximate (HNSW) index: a collection holds a few
thousand rows, an exact scan is milliseconds, and a key filter can never cut the result below k as a filtered
approximate scan can. Score semantics are Chroma's cosine space: distance = embedding <=> query (1 - cosine).

The address comes from the caller's VectorStoreConfig; nothing here reads the environment or .env.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from typing import Any

from kosis_rag.schemas.resources import VectorStoreConfig

MAX_BATCH = 1000  # rows per upsert call (Chroma's client reported its own limit; the build code slices by it)

SCHEMA = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS vector_collection (
        name text PRIMARY KEY,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS vector_item (
        collection text NOT NULL REFERENCES vector_collection(name) ON DELETE CASCADE,
        id text NOT NULL,
        key text,
        document text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        embedding vector NOT NULL,
        PRIMARY KEY (collection, id))""",
    "CREATE INDEX IF NOT EXISTS vector_item_key ON vector_item (collection, key)",
)


class CollectionNotFound(ValueError):
    """No collection of that name in the store (Chroma raised NotFoundError here)."""


def plain_url(url: str) -> str:
    """psycopg wants postgresql://; the backend's DATABASE_URL is the SQLAlchemy form postgresql+psycopg://."""
    return re.sub(r"^postgres(ql)?\+\w+://", "postgresql://", url)


def safe_url(url: str) -> str:
    """The URL without its password, for messages."""
    return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", plain_url(url))


def key_filter(where: dict | None) -> list[str] | None:
    """The table keys of a `{"key": {"$in": [...]}}` filter; None for no filter. Any other filter is refused: the
    search decides its conditions on the registry and sends only the passing keys (shared.filters)."""
    if not where:
        return None
    if set(where) == {"key"} and isinstance(where["key"], dict) and set(where["key"]) == {"$in"}:
        return [str(k) for k in where["key"]["$in"]]
    raise ValueError(f"unsupported filter {where!r}: the vector store takes only {{'key': {{'$in': [...]}}}}")


def as_array(value):
    """A stored vector as a float32 numpy array (what the Chroma client returned); pgvector may hand back a Vector."""
    import numpy as np

    return np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value, dtype=np.float32)


def query_sql(n_results: int, keys: list[str] | None) -> tuple[str, list]:
    """The nearest-rows statement of one query and its parameters after the query vector (collection, [keys], n).
    Ties are broken by id so a reload gives the same order."""
    where = "collection = %s" + (" AND key = ANY(%s)" if keys is not None else "")
    sql = (f"SELECT id, document, metadata, embedding <=> %s AS distance FROM vector_item WHERE {where} "
           "ORDER BY distance, id LIMIT %s")
    return sql, ([keys] if keys is not None else []) + [int(n_results)]


class VectorCollection:
    """One collection; the subset of chromadb.Collection the search and the builds call."""

    def __init__(self, client: VectorClient, name: str, metadata: dict | None):
        self._client, self.name, self._metadata = client, name, metadata

    @property
    def metadata(self) -> dict:
        return dict(self._metadata or {})

    def count(self) -> int:
        return self._client._one("SELECT count(*) FROM vector_item WHERE collection = %s", [self.name])[0]

    def modify(self, metadata: dict | None = None, name: str | None = None) -> None:
        if name is not None:
            raise ValueError("renaming a collection is not supported")
        if metadata is not None:
            from psycopg.types.json import Jsonb

            self._client._run("UPDATE vector_collection SET metadata = %s WHERE name = %s", [Jsonb(metadata), self.name])
            self._metadata = dict(metadata)

    def _check_dim(self, vectors: Sequence[Sequence[float]]) -> None:
        dim = (self._metadata or {}).get("dim")
        bad = [len(v) for v in vectors if dim is not None and len(v) != int(dim)]
        if bad:
            raise ValueError(f"collection {self.name} holds {dim}-dimension vectors, got {bad[0]}")

    def upsert(self, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[dict],
               embeddings: Sequence[Sequence[float]]) -> None:
        import numpy as np
        from psycopg.types.json import Jsonb

        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, documents, metadatas and embeddings must have the same length")
        self._check_dim(embeddings)
        rows = [(self.name, str(i), (m or {}).get("key"), d, Jsonb(m or {}), np.asarray(e, dtype=np.float32))
                for i, d, m, e in zip(ids, documents, metadatas, embeddings)]
        self._client._many(
            "INSERT INTO vector_item (collection, id, key, document, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (collection, id) DO UPDATE SET "
            "key = EXCLUDED.key, document = EXCLUDED.document, metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding", rows)

    add = upsert

    def delete(self, ids: Sequence[str] | None = None) -> None:
        if ids:
            self._client._run("DELETE FROM vector_item WHERE collection = %s AND id = ANY(%s)",
                              [self.name, [str(i) for i in ids]])

    def get(self, ids: Sequence[str] | None = None, where: dict | None = None,
            include: Sequence[str] = ("documents", "metadatas"), limit: int | None = None,
            offset: int | None = None) -> dict[str, Any]:
        """Rows in id order: {"ids", "documents", "metadatas", "embeddings"} (only the included lists are filled)."""
        include = set(include)
        keys = key_filter(where)
        cond, params = "collection = %s", [self.name]
        if ids is not None:
            cond, params = cond + " AND id = ANY(%s)", params + [[str(i) for i in ids]]
        if keys is not None:
            cond, params = cond + " AND key = ANY(%s)", params + [keys]
        sql = f"SELECT id, document, metadata, {'embedding' if 'embeddings' in include else 'NULL'} " \
              f"FROM vector_item WHERE {cond} ORDER BY id"
        if limit is not None:
            sql, params = sql + " LIMIT %s", params + [int(limit)]
        if offset:
            sql, params = sql + " OFFSET %s", params + [int(offset)]
        rows = self._client._all(sql, params)
        return {"ids": [r[0] for r in rows],
                "documents": [r[1] for r in rows] if "documents" in include else None,
                "metadatas": [r[2] for r in rows] if "metadatas" in include else None,
                "embeddings": [as_array(r[3]) for r in rows] if "embeddings" in include else None}

    def query(self, query_embeddings: Sequence[Sequence[float]], n_results: int = 10,
              include: Sequence[str] = ("documents", "metadatas", "distances"),
              where: dict | None = None) -> dict[str, list[list]]:
        """Nearest rows per query vector, closest first: {"ids", "documents", "metadatas", "distances"}, each a list
        per query vector (Chroma's shape). Fewer than n_results only when fewer rows match."""
        import numpy as np

        self._check_dim(query_embeddings)
        sql, tail = query_sql(n_results, key_filter(where))
        out: dict[str, list[list]] = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        for vector in query_embeddings:
            q = np.asarray(vector, dtype=np.float32)
            rows = self._client._all(sql, [q, self.name] + tail)
            out["ids"].append([r[0] for r in rows])
            out["documents"].append([r[1] for r in rows])
            out["metadatas"].append([r[2] for r in rows])
            out["distances"].append([float(r[3]) for r in rows])
        return out


class VectorClient:
    """One PostgreSQL connection (autocommit, pgvector types registered); a dropped connection is reopened once."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._lock = threading.Lock()
        self._conn = None
        self._connect()

    def _connect(self) -> None:
        import psycopg

        try:
            conn = psycopg.connect(plain_url(self.config.url), autocommit=True, connect_timeout=5)
        except psycopg.OperationalError as exc:
            raise ConnectionError(
                f"PostgreSQL is not reachable at {safe_url(self.config.url)}. Start Docker Desktop, then run: "
                "docker compose up -d postgres") from exc
        ensure_schema(conn)
        from pgvector.psycopg import register_vector

        register_vector(conn)
        self._conn = conn

    def _exec(self, fn):
        import psycopg

        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._conn is None or self._conn.closed:
                        self._connect()
                    with self._conn.cursor() as cur:
                        return fn(cur)
                except psycopg.OperationalError:
                    self._conn = None
                    if attempt:
                        raise

    def _run(self, sql: str, params: Sequence = ()) -> None:
        self._exec(lambda cur: cur.execute(sql, params))

    def _many(self, sql: str, rows: list) -> None:
        self._exec(lambda cur: cur.executemany(sql, rows))

    def _one(self, sql: str, params: Sequence = ()):
        return self._exec(lambda cur: cur.execute(sql, params).fetchone())

    def _all(self, sql: str, params: Sequence = ()) -> list:
        return self._exec(lambda cur: cur.execute(sql, params).fetchall())

    def get_max_batch_size(self) -> int:
        return MAX_BATCH

    def get_collection(self, name: str, embedding_function: Any = None) -> VectorCollection:
        row = self._one("SELECT metadata FROM vector_collection WHERE name = %s", [name])
        if row is None:
            raise CollectionNotFound(f"collection {name} is not in the vector store ({safe_url(self.config.url)})")
        return VectorCollection(self, name, row[0])

    def get_or_create_collection(self, name: str, embedding_function: Any = None,
                                 metadata: dict | None = None) -> VectorCollection:
        """Like Chroma: an existing collection keeps its metadata (the caller refreshes it with modify)."""
        from psycopg.types.json import Jsonb

        self._run("INSERT INTO vector_collection (name, metadata) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                  [name, Jsonb(metadata or {})])
        return self.get_collection(name)

    def list_collections(self) -> list[VectorCollection]:
        return [VectorCollection(self, n, m) for n, m in
                self._all("SELECT name, metadata FROM vector_collection ORDER BY name")]

    def delete_collection(self, name: str) -> None:
        self._run("DELETE FROM vector_collection WHERE name = %s", [name])


def ensure_schema(conn) -> None:
    """Create the extension and the two tables when missing (the backend's alembic migration makes the same ones
    first; a build or test run without it still finds them)."""
    with conn.cursor() as cur:
        for statement in SCHEMA:
            cur.execute(statement)


def vector_client(config: VectorStoreConfig) -> VectorClient:
    """Client for the configured database. Connects on creation, so it fails fast when the server is down."""
    return VectorClient(config)
