"""T2-23, T2-41 S1: the Chroma address comes from the caller's ChromaConfig; no server is contacted here (verify.sh
stays offline). The environment is read by the callers (scripts, until S4 the src callers), not by adapters.chroma."""

import pytest

from kosis_rag.adapters import chroma
from kosis_rag.schemas.resources import ChromaConfig

offline = pytest.mark.offline


@offline
def test_config_defaults_to_the_host_port_of_the_compose_service():
    config = ChromaConfig()
    assert (config.host, config.port, config.collection) == ("localhost", 8001, None)


@offline
def test_client_connects_to_the_configured_address_as_compose_sets_it_for_the_backend(monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    seen = {}
    monkeypatch.setattr(chromadb, "HttpClient", lambda host, port, settings: seen.update(host=host, port=port) or "client")
    assert chroma.chroma_client(ChromaConfig(host="chroma", port=8000)) == "client"
    assert seen == {"host": "chroma", "port": 8000}


@offline
def test_client_says_how_to_start_the_server_when_it_is_down(monkeypatch):
    chromadb = pytest.importorskip("chromadb")

    def refuse(**_):
        raise ValueError("Could not connect to a Chroma server")

    monkeypatch.setattr(chromadb, "HttpClient", refuse)
    with pytest.raises(ConnectionError, match="localhost:8001.*docker compose up -d chroma"):
        chroma.chroma_client(ChromaConfig())


@offline
def test_adapter_reads_no_environment(monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    monkeypatch.setenv("CHROMA_HOST", "elsewhere")
    monkeypatch.setenv("CHROMA_PORT", "9999")
    seen = {}
    monkeypatch.setattr(chromadb, "HttpClient", lambda host, port, settings: seen.update(host=host, port=port))
    chroma.chroma_client(ChromaConfig())
    assert seen == {"host": "localhost", "port": 8001}
