from fastapi import FastAPI

app = FastAPI(title="KOSIS RAG API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
