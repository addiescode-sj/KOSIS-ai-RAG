from fastapi.testclient import TestClient

from backend.app.main import app


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_phase1_contract_importable():
    # Phase 1 schemas come from the vendor/phase1 submodule; this fails if it is not checked out.
    from kosis_ai.schemas import StatAnswer

    assert StatAnswer
