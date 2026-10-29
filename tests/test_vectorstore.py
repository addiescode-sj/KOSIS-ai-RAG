"""T2-23: the Chroma address comes from the environment; no server is contacted here (verify.sh stays offline)."""

import pytest

from kosis_rag.vectorstore import chroma_address

offline = pytest.mark.offline


@offline
def test_address_defaults_to_the_host_port_of_the_compose_service(monkeypatch):
    monkeypatch.setattr("kosis_rag.vectorstore.load_dotenv", lambda: None)
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.delenv("CHROMA_PORT", raising=False)
    assert chroma_address() == ("localhost", 8001)


@offline
def test_address_follows_the_environment_as_compose_sets_it_for_the_backend(monkeypatch):
    monkeypatch.setattr("kosis_rag.vectorstore.load_dotenv", lambda: None)
    monkeypatch.setenv("CHROMA_HOST", "chroma")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    assert chroma_address() == ("chroma", 8000)
