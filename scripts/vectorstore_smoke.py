"""Check the vector store (PostgreSQL + pgvector, T2-44; was the T2-23 Chroma check): connect, write, restart the
container, read back, clean up.

Run from the repo root after `docker compose up -d postgres` (address: DATABASE_URL in the shell, default the compose
postgres on localhost:5432):
    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/vectorstore_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from kosis_rag.adapters.vectorstore import safe_url, vector_client
from kosis_rag.schemas.resources import VectorStoreConfig

COLLECTION = "t2_44_smoke"


def _wait(config: VectorStoreConfig, attempts: int = 30):
    for _ in range(attempts):
        try:
            return vector_client(config)
        except ConnectionError:
            time.sleep(1)
    return vector_client(config)  # raises with the start-up hint


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    config = VectorStoreConfig(**({"url": url} if url else {}))
    client = _wait(config, 1)
    print(f"connected to {safe_url(config.url)}: {len(client.list_collections())} collections")
    client.get_or_create_collection(COLLECTION, metadata={"dim": 3}).upsert(
        ids=["101_DT_TEST"], documents=["smoke test table"], embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"key": "101_DT_TEST", "tbl_id": "DT_TEST", "start": 2015}],
    )
    try:
        subprocess.run(["docker", "compose", "restart", "postgres"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"could not restart the postgres service ({exc}); is the docker CLI on PATH?", file=sys.stderr)
        return 1
    time.sleep(2)
    got = _wait(config).get_collection(COLLECTION).get(ids=["101_DT_TEST"], include=["metadatas"])
    _wait(config).delete_collection(COLLECTION)
    survived = got["ids"] == ["101_DT_TEST"] and got["metadatas"][0]["tbl_id"] == "DT_TEST"
    print("after restart:", "data kept" if survived else f"data lost ({got})")
    return 0 if survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
