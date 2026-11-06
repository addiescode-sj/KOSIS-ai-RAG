"""KOSIS RAG API (T2-37): the LangGraph search nodes, one endpoint each, plus the whole search as JSON and as SSE.

Search logic lives in src/kosis_rag (graph.nodes_api.make_nodes, graph.search.graph_nodes, ground_docs), the answer
(refuse | guide) in backend/app/respond (T2-41 S5: respond.graph.build_answer_graph, api_nodes, stream_answer); this
module only exposes them. Settings come from the shell environment (backend/app/config.py; no .env is read, T2-41 S4).
Node endpoints are stateless: they take a SearchState and return it with that node's fields written, so a client can
chain parse -> retrieve -> ... -> answer itself or re-run one node on an edited state. Tables in a request are
re-attached to the server's records (ground_docs), so a client cannot invent a table or a citation.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import KW_ONLY, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from kosis_rag.graph.nodes_api import ground_docs
from kosis_rag.llm.config import LLMConfig, LLMKeyMissing
from kosis_rag.schemas.io import AskArgs, SearchResult
from kosis_rag.schemas.resources import SearchResources

from backend.app.respond.graph import api_nodes, build_answer_graph, run_answer, stream_answer
from backend.app.respond.node import GuideClient, Node, as_guide_client
from backend.app.respond.state import SearchState

LOG_DIR = Path("logs/api")  # gitignored; one JSON line per search, written after the response (BackgroundTasks)
QUESTION = {"question": "2023년 시도별 출생아 수", "k": 5}

app = FastAPI(
    title="KOSIS RAG API",
    version="0.2.0",
    description=(
        "국가데이터처(KOSIS) 통계표 안내 검색 API.\n\n"
        "검색은 LangGraph 노드 `parse → retrieve → [rerank1] → [rerank2] → gate → refuse | guide` 로 돈다.\n"
        "그래프 안에서 retrieve는 `condition → [embed] → retrieve_dense ∥ retrieve_bm25 ∥ [retrieve_hq] ∥ "
        "[retrieve_hq_bm25] → fuse` 병렬 노드로 나뉘고(T2-39), `/nodes/retrieve`는 이 단계를 한 번에 돈다. "
        "`/search/stream`의 노드 이벤트는 그래프 노드 이름으로 온다.\n"
        "- **nodes**: 노드 하나씩 부른다. 요청·응답 모두 `SearchState`라 앞 노드의 응답을 다음 노드에 그대로 보낸다.\n"
        "- **search**: 전체 그래프를 한 번에 (`/search` JSON, `/search/stream` SSE 노드 이벤트).\n\n"
        "표 ID·출처는 서버의 통계표 레지스트리에서만 나온다. 맞는 표가 없으면 거절 답(`no_answer=true`)을 준다."
    ),
    openapi_tags=[
        {"name": "nodes", "description": "검색 그래프의 노드 하나를 실행한다 (무상태: SearchState in → SearchState out)."},
        {"name": "search", "description": "검색 그래프 전체 실행. 응답 뒤 노드별 지연을 `logs/api/`에 기록한다."},
        {"name": "ops", "description": "상태 확인."},
    ],
)


@app.exception_handler(LLMKeyMissing)
async def llm_key_missing(request: Request, exc: LLMKeyMissing) -> JSONResponse:
    """A configured LLM node has no API key (T2-41 S3: raised at the node's entry): 503, not an unhandled 500."""
    return JSONResponse(status_code=503, content={"detail": f"LLM key missing: {exc}"})


@dataclass
class Runtime:
    """What the endpoints share: the loaded resources, the guide LLM, and one compiled graph per k.

    The guide client is made on first use (T2-41 S5: respond.node.GuideClient, built from guide_llm through
    kosis_rag.llm.providers and kept here for every graph and node set): parse ... gate run without a guide key, and
    a missing key is LLMKeyMissing (503) when guide runs. `client` is a ready guide client (tests: MockClient).
    """

    resources: SearchResources
    _: KW_ONLY
    guide_llm: LLMConfig | None = None  # backend.app.config.guide_llm_from_env(); None = no guide LLM
    client: Any = None  # a ready guide client (LLMClient | BaseChatModel); None = make one from guide_llm
    graphs: dict[int, Any] = field(default_factory=dict)
    guide: GuideClient = field(init=False)

    def __post_init__(self) -> None:
        self.guide = as_guide_client(self.client if self.client is not None else self.guide_llm)

    def guide_client(self) -> Any:
        """The guide LLM client (made once on the first call). Side effects: the first call builds it from
        guide_llm; LLMKeyMissing when guide_llm is None or has no key (nothing is kept, the next call checks again)."""
        return self.guide.get()

    def graph(self, k: int):
        if k not in self.graphs:  # ponytail: one graph per k (k <= 20), compiling is cheap next to the resources
            self.graphs[k] = build_answer_graph(self.resources, self.guide, k=k)
        return self.graphs[k]

    def nodes(self, k: int) -> dict[str, Node]:
        return api_nodes(self.resources, self.guide, k=k)

    def k(self, requested: int | None) -> int:
        """The candidate budget: the request's k, else the server's ANSWER_K (T2-39)."""
        return requested or self.resources.options.answer_k


_runtime: Runtime | None = None
_lock = threading.Lock()


def get_runtime() -> Runtime:
    """Load the resources on first use (not at startup, so /health works without Chroma or data/).

    Out: the shared Runtime. Side effects: the first call reads the shell environment (backend/app/config.py, no
    .env), reads tables.jsonl, connects to Chroma, loads the models; a failure is a 503 and the next request tries
    again. The guide LLM is not made here (Runtime.guide, on the first guide run).
    """
    global _runtime
    with _lock:
        if _runtime is None:
            from backend.app.config import search_config_from_env
            from kosis_rag.adapters.loader import load_resources

            try:
                cfg = search_config_from_env()
                resources = load_resources(spec=cfg.spec, options=cfg.options, llm=cfg.stage2_llm)
                _runtime = Runtime(resources=resources, guide_llm=cfg.guide_llm)
            except Exception as e:  # noqa: BLE001 - any load failure means the search is unavailable
                raise HTTPException(503, f"search resources not available: {e}") from e
        return _runtime


def _log(state: SearchState) -> None:
    """Append one line per search: question, per-node seconds, route, top keys, answer origin (BackgroundTasks)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": state.question, "route": state.route,
        "seconds": {name: round(t["seconds"], 4) for name, t in state.trace.items()},
        "top": [d.metadata.get("key") for d in state.docs[:5]],
        "answer_origin": state.result.answer_origin if state.result else None,
    }
    with (LOG_DIR / f"{date.today()}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


# --- ops ---

@app.get("/health", tags=["ops"], summary="서버 상태")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --- nodes ---

class ParseRequest(BaseModel):
    question: str = Field(min_length=1, description="사용자 질문 (한국어)", examples=["2023년 시도별 출생아 수"])


class NodeList(BaseModel):
    nodes: list[str] = Field(description="이 서버에서 켜진 노드, 실행 순서. answer는 gate의 route에 따라 refuse 또는 guide")


K = Query(None, ge=1, le=20, description="안내 LLM(gate·answer)에 넘길 후보 표 수 docs[:k]. 생략하면 서버 설정 ANSWER_K(기본 5). "
          "답변에 보여 줄 표는 이 수가 아니라 LLM이 고른 표(result.picked)다")

NODE_DOCS = {
    "retrieve": ("하이브리드 검색", "질문 조건(연도·주기 등)으로 거른 뒤 dense + BM25(+ 켜져 있으면 가상 질문 색인 HQ, T2-39)를 "
                                "RRF로 합친다. `docs`·`removed`·`relaxed`를 쓴다. 그래프의 condition → [embed] → "
                                "retrieve_* → fuse를 한 번에 돌리며, 목록별 결과 `runs`·`run_removed`와 `table_filter`도 "
                                "쓴다. 요청의 이 필드들과 `query_vecs`는 무시한다."),
    "rerank1": ("1차 리랭커 (cross-encoder)", "상위 rerank_n을 cross-encoder 점수로 다시 정렬한다. 꺼져 있으면 409."),
    "rerank2": ("2차 리랭커 (LLM)", "LLM이 상위 stage2_k 중에서 고른다. 아무것도 안 고르면 gate가 거절한다. 꺼져 있으면 409."),
    "gate": ("거절 판정", "상위 k로 `refuse` / `guide`를 정해 `route`·`refusal_reason`을 쓴다."),
}


@app.get("/nodes", tags=["nodes"], summary="켜진 노드 목록", response_model=NodeList)
def list_nodes(rt: Runtime = Depends(get_runtime)) -> NodeList:
    return NodeList(nodes=[name for name in rt.nodes(rt.k(None)) if name not in ("refuse", "guide")] + ["answer"])


async def _run_node(rt: Runtime, name: str, state: SearchState, k: int | None) -> SearchState:
    """Ground the request's tables, run one node in the threadpool, return the state with its update applied."""
    nodes = rt.nodes(rt.k(k))
    if name not in nodes:
        raise HTTPException(409, f"node {name!r} is off in this server's settings; on: {list(nodes)}")
    try:
        state = state.model_copy(update={"docs": ground_docs(state.docs, rt.resources.records)})
    except KeyError as e:
        raise HTTPException(422, str(e.args[0])) from e
    update = await run_in_threadpool(nodes[name], state)
    return state.model_copy(update=update)


@app.post("/nodes/parse", tags=["nodes"], summary="질의 조건 파싱", response_model=SearchState,
          description="질문에서 연도·주기·지역·세는 대상 등 검색 조건을 규칙으로 읽어 `query`를 쓴다. LLM 호출 없음.")
async def node_parse(req: ParseRequest, k: int | None = K, rt: Runtime = Depends(get_runtime)) -> SearchState:
    return await _run_node(rt, "parse", SearchState(question=req.question), k)


def _node_route(name: str, summary: str, description: str) -> None:
    async def endpoint(state: SearchState, k: int | None = K, rt: Runtime = Depends(get_runtime)) -> SearchState:
        return await _run_node(rt, name, state, k)

    app.post(f"/nodes/{name}", tags=["nodes"], summary=summary, description=description, response_model=SearchState,
             operation_id=f"node_{name}")(endpoint)


for _name, (_summary, _description) in NODE_DOCS.items():
    _node_route(_name, _summary, _description)


@app.post("/nodes/answer", tags=["nodes"], summary="답변 (거절 또는 안내)", response_model=SearchState,
          description="gate가 쓴 `route`로 refuse(코드 문장, LLM 없음) 또는 guide(LLM 안내 + 코드 검증·출처)를 실행해 "
                      "`result`를 쓴다. route가 없거나, guide인데 docs가 비면 422.")
async def node_answer(state: SearchState, k: int | None = K, rt: Runtime = Depends(get_runtime)) -> SearchState:
    if state.route is None:
        raise HTTPException(422, "route is not set: call /nodes/gate first")
    if state.route == "guide" and not state.docs:
        raise HTTPException(422, "route 'guide' needs at least one table in docs")
    return await _run_node(rt, state.route, state, k)


# --- search ---

@app.post("/search", tags=["search"], summary="검색 + 답변 (JSON)", response_model=SearchResult,
          openapi_extra={"requestBody": {"content": {"application/json": {"example": QUESTION}}}})
async def search(args: AskArgs, background: BackgroundTasks, rt: Runtime = Depends(get_runtime)) -> SearchResult:
    """전체 그래프를 돌려 `SearchResult`를 돌려준다. `picked` = 답변이 고른 표(출처도 이것), `candidates` = LLM에 넘긴 후보.
    노드별 지연은 응답 뒤 기록한다."""
    state = await run_in_threadpool(run_answer, rt.graph(rt.k(args.k)), args.question)
    background.add_task(_log, state)
    return state.result


@app.post("/search/stream", tags=["search"], summary="검색 + 답변 (SSE 노드 이벤트)",
          response_class=EventSourceResponse,
          openapi_extra={"requestBody": {"content": {"application/json": {"example": QUESTION}}}})
def search_stream(args: AskArgs, background: BackgroundTasks,
                  rt: Runtime = Depends(get_runtime)) -> Iterator[ServerSentEvent]:
    """노드가 끝날 때마다 `event: <노드 이름>` (data = 그 노드의 trace: seconds 등)을 보내고,
    guide가 안내 문장을 쓰는 동안 `event: token` (data = `{"delta": "..."}`)을 보낸다 — 검증(후보에 없는 표 ID·연도)을
    통과한 앞부분만 단어 단위로 흐르고, `NO_MATCH`는 흐르지 않는다. 마지막에 `event: result`
    (data = `SearchResult`, `/search`와 같은 값, 출처 포함 최종 문장)를 보낸다.

    예: `curl -N -X POST localhost:8000/search/stream -H 'content-type: application/json' -d '{"question": "..."}'`
    """
    try:
        for name, payload in stream_answer(rt.graph(rt.k(args.k)), args.question):
            if name == "result":
                background.add_task(_log, payload)
                yield ServerSentEvent(event="result", data=payload.result)
            else:
                yield ServerSentEvent(event=name, data=payload)
    except Exception as e:  # noqa: BLE001 - the status line is already sent; say it in the stream instead
        yield ServerSentEvent(event="error", data={"type": type(e).__name__, "detail": str(e)})
