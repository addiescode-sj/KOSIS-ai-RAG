"""Check the shared ChromaDB: heartbeat, write, restart the container, read back, clean up (T2-23).

Run from the repo root after `docker compose up -d chroma` (address: CHROMA_HOST / CHROMA_PORT in the shell,
default localhost:8001):
    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/chroma_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from kosis_rag.adapters.chroma import chroma_client
from kosis_rag.schemas.resources import ChromaConfig

COLLECTION = "t2_23_smoke"


def _wait(config: ChromaConfig, attempts: int = 30):
    for _ in range(attempts):
        try:
            return chroma_client(config)
        except ConnectionError:
            time.sleep(1)
    return chroma_client(config)  # raises with the start-up hint


def main() -> int:
    config = ChromaConfig(host=os.environ.get("CHROMA_HOST", "localhost"), port=int(os.environ.get("CHROMA_PORT", 8001)))
    client = _wait(config, 1)
    print(f"heartbeat ok at {config.host}:{config.port}: {client.heartbeat()}")
    client.get_or_create_collection(COLLECTION).upsert(
        ids=["101_DT_TEST"], documents=["smoke test table"], embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"tbl_id": "DT_TEST", "start": 2015}],
    )
    try:
        subprocess.run(["docker", "compose", "restart", "chroma"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"could not restart the chroma service ({exc}); is the docker CLI on PATH?", file=sys.stderr)
        return 1
    got = _wait(config).get_collection(COLLECTION).get(ids=["101_DT_TEST"], include=["metadatas"])
    _wait(config).delete_collection(COLLECTION)
    survived = got["ids"] == ["101_DT_TEST"] and got["metadatas"][0]["tbl_id"] == "DT_TEST"
    print("after restart:", "data kept" if survived else f"data lost ({got})")
    return 0 if survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
