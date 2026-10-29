"""Check the shared ChromaDB: heartbeat, write, restart the container, read back, clean up (T2-23).

Run from the repo root after `docker compose up -d chroma`:
    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python scripts/chroma_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
import time

from kosis_rag.vectorstore import chroma_address, chroma_client

COLLECTION = "t2_23_smoke"


def _wait(attempts: int = 30):
    for _ in range(attempts):
        try:
            return chroma_client()
        except ConnectionError:
            time.sleep(1)
    return chroma_client()  # raises with the start-up hint


def main() -> int:
    client = _wait(1)
    print(f"heartbeat ok at {':'.join(map(str, chroma_address()))}: {client.heartbeat()}")
    client.get_or_create_collection(COLLECTION).upsert(
        ids=["101_DT_TEST"], documents=["smoke test table"], embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"tbl_id": "DT_TEST", "start": 2015}],
    )
    try:
        subprocess.run(["docker", "compose", "restart", "chroma"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"could not restart the chroma service ({exc}); is the docker CLI on PATH?", file=sys.stderr)
        return 1
    got = _wait().get_collection(COLLECTION).get(ids=["101_DT_TEST"], include=["metadatas"])
    _wait().delete_collection(COLLECTION)
    survived = got["ids"] == ["101_DT_TEST"] and got["metadatas"][0]["tbl_id"] == "DT_TEST"
    print("after restart:", "data kept" if survived else f"data lost ({got})")
    return 0 if survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
