"""The one way stage 2 code reaches the shared ChromaDB (docker compose service `chroma`, T2-23; T2-41 S1: from
vectorstore.py).

The address comes from the caller's ChromaConfig; nothing here reads the environment or .env. On the host the
defaults match `docker compose up -d chroma` (localhost:8001); inside compose the backend uses chroma:8000.
"""

from __future__ import annotations

from kosis_rag.schemas.resources import ChromaConfig


def chroma_client(config: ChromaConfig):
    """HTTP client for the shared server. Connects on creation, so it fails fast when the server is down."""
    import chromadb  # chromadb-client; imported here so offline code paths do not need it
    from chromadb.config import Settings

    host, port = config.host, config.port
    try:
        return chromadb.HttpClient(host=host, port=port, settings=Settings(anonymized_telemetry=False))
    except Exception as exc:  # the client raises plain exceptions when it cannot connect
        raise ConnectionError(
            f"ChromaDB is not reachable at {host}:{port}. Start Docker Desktop, then run: docker compose up -d chroma"
        ) from exc
