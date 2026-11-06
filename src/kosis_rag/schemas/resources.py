"""Settings objects the search module is built from (T2-41). S1 holds only ChromaConfig; SearchOptions, IndexSpec and
SearchResources join at S4 (plans/T2-41.md, 입출력 계약)."""

from __future__ import annotations

from pydantic import BaseModel


class ChromaConfig(BaseModel):
    """Where the shared ChromaDB server is. Defaults match `docker compose up -d chroma` on the host (port 8001);
    inside compose the backend uses host "chroma", port 8000. The caller builds it (no .env read here)."""

    host: str = "localhost"
    port: int = 8001
    collection: str | None = None  # None: the caller names the collection
