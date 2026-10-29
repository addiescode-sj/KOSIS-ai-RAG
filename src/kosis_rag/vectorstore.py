"""The one way stage 2 code reaches the shared ChromaDB (docker compose service `chroma`, T2-23).

On the host (notebooks, scripts, tests) the address comes from .env: CHROMA_HOST=localhost, CHROMA_PORT=8001.
Inside compose the backend gets CHROMA_HOST=chroma, CHROMA_PORT=8000 from docker-compose.yml.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def chroma_address() -> tuple[str, int]:
    """(host, port) from the environment or .env; defaults match `docker compose up -d chroma` on the host."""
    load_dotenv()
    return os.getenv("CHROMA_HOST") or "localhost", int(os.getenv("CHROMA_PORT") or 8001)


def chroma_client():
    """HTTP client for the shared server. Connects on creation, so it fails fast when the server is down."""
    import chromadb  # chromadb-client; imported here so offline code paths do not need it
    from chromadb.config import Settings

    host, port = chroma_address()
    try:
        return chromadb.HttpClient(host=host, port=port, settings=Settings(anonymized_telemetry=False))
    except Exception as exc:  # the client raises plain exceptions when it cannot connect
        raise ConnectionError(
            f"ChromaDB is not reachable at {host}:{port}. Start Docker Desktop, then run: docker compose up -d chroma"
        ) from exc
