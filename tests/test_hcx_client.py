"""T2-41 S3: the HCXClient copy (llm/hcx.py) against a local fake CLOVA Studio server (no network, dummy key).

Phase 1 tests/test_hcx.py's cases, with an LLMConfig instead of Phase 1 Settings + HCX_API_KEY in the environment
(the copy's constructor is the only changed part, plans/T2-41.md S3 risk).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kosis_rag.llm.config import LLMConfig, LLMKeyMissing
from kosis_rag.llm.hcx import HCXClient
from kosis_rag.llm.providers import build_client

pytestmark = pytest.mark.offline

KEY = "nv-dummy"


class FakeServer:
    """Answers every POST with `status` / `body` / `headers` (changeable between requests) and records requests."""

    def __init__(self, status=200, body=None, headers=None, delay=0.0):
        self.status, self.body, self.headers, self.delay = status, body, headers or {}, delay
        self.requests: list[dict] = []
        self.request_headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                outer.request_headers.append(dict(self.headers))
                if outer.delay:
                    threading.Event().wait(outer.delay)
                payload = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                for name, value in outer.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server_factory(monkeypatch):
    for name in ("NO_PROXY", "no_proxy"):  # the fake server is local: never through a proxy
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    servers = []

    def make(**kwargs):
        servers.append(FakeServer(**kwargs))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


def no_sleep(_):
    pass


def _config(**kw):
    return LLMConfig(**{"provider": "hcx", "model": "HCX-007", "thinking": "none", "api_key": KEY, **kw})


def _client(base_url, sleep=no_sleep, **kw):
    return HCXClient(_config(**kw), sleep=sleep, base_url=base_url)


def ok_body(text="2023년 여자 인구는 100명입니다."):
    return {
        "status": {"code": "20000", "message": "OK"},
        "result": {
            "message": {"role": "assistant", "content": text},
            "finishReason": "stop",
            "usage": {"promptTokens": 120, "completionTokens": 15, "totalTokens": 135},
        },
    }


def test_reasoning_model_request_shape(server_factory):
    srv = server_factory(body=ok_body())
    result = _client(srv.base_url).generate("sys", "여자 인구는?")
    assert result.status == "ok" and "100명" in result.text and result.attempts == 1
    req = srv.requests[0]
    assert req["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "여자 인구는?"}]
    assert req["temperature"] == 0 and req["maxCompletionTokens"] == 1024
    assert req["thinking"] == {"effort": "none"} and "maxTokens" not in req
    assert srv.request_headers[0]["Authorization"] == f"Bearer {KEY}"
    assert _client(srv.base_url).url == srv.base_url + "/v3/chat-completions/HCX-007"


def test_light_model_uses_max_tokens_without_thinking(server_factory):
    srv = server_factory(body=ok_body())
    client = _client(srv.base_url, model="HCX-DASH-002", max_tokens=512)
    assert client.generate("sys", "user").usage["totalTokens"] == 135
    assert srv.requests[0]["maxTokens"] == 512 and "thinking" not in srv.requests[0]


def test_retries_rate_limit_then_succeeds(server_factory):
    srv = server_factory(status=429, body={"status": {"code": "42901", "message": "Too many requests"}})
    delays = []

    def fake_sleep(delay):
        delays.append(delay)
        srv.status, srv.body = 200, ok_body("ok")

    result = _client(srv.base_url, sleep=fake_sleep).generate("s", "u")
    assert result.status == "ok" and result.text == "ok" and result.attempts == 2
    assert delays == [HCXClient.RETRY_DELAYS[0]] and len(srv.requests) == 2


def test_retry_waits_for_longer_retry_after(server_factory):
    srv = server_factory(status=429, body={"status": {"code": "42901"}}, headers={"Retry-After": "40"})
    delays = []
    _client(srv.base_url, sleep=delays.append).generate("s", "u")
    assert delays == [40.0, 40.0, 40.0, 60.0]


def test_errors_become_statuses(server_factory):
    busy = server_factory(status=503, body={"status": {"code": "50000", "message": "busy"}})
    result = _client(busy.base_url).generate("s", "u")
    assert result.status == "api_error" and "HTTP 503" in result.error
    assert len(busy.requests) == 1 + len(HCXClient.RETRY_DELAYS)

    bad = server_factory(status=401, body={"status": {"code": "40100", "message": "Unauthorized"}})
    assert "HTTP 401" in _client(bad.base_url).generate("s", "u").error
    assert len(bad.requests) == 1  # auth errors are not retried

    long = server_factory(status=400, body={"status": {"code": "40003", "message": "Context length exceeded."}})
    assert _client(long.base_url).generate("s", "u").status == "input_too_long"

    empty = server_factory(body=ok_body(""))
    assert "empty response" in _client(empty.base_url).generate("s", "u").error


def test_timeout_and_connection_failure(server_factory):
    slow = server_factory(body=ok_body(), delay=1.0)
    assert _client(slow.base_url, timeout=0.2).generate("s", "u").status == "timeout"
    assert _client("http://127.0.0.1:9").generate("s", "u").status == "api_error"


def test_key_is_redacted_from_what_the_observer_sees(server_factory):
    srv = server_factory(body=ok_body(f"echo {KEY}"))
    events = []
    result = HCXClient(_config(), sleep=no_sleep, observer=events.append, base_url=srv.base_url).generate("s", "u")
    assert KEY not in result.text and "[REDACTED]" in result.text
    assert [e["type"] for e in events] == ["http_request", "http_response"] and KEY not in json.dumps(events)


def test_missing_key_or_model_is_a_config_error():
    with pytest.raises(LLMKeyMissing, match="provider='hcx'"):
        build_client(_config(api_key=None))
    with pytest.raises(ValueError, match="model"):
        HCXClient(_config(model="TBD"))
