"""States and I/O of the offline build graphs (T2-41 S6a; the hq part from the old hq/schemas.py, hq/graph.py).

    TablesBuildState      graph/index_build.build_tables_graph     read_parquet -> build_records -> write_jsonl
    SigunguBuildState     graph/index_build.build_sigungu_graph    read_parquet -> build_sigungu -> write_json
    IndexBuildState       graph/index_build.build_chroma_graph     read_records -> build_docs -> embed -> load | write_build
    QuestionsIndexState   graph/index_build.build_questions_graph  read_questions -> embed -> load
    HqState               graph/hq_build.build_table_graph         one table: load -> prompt -> generate -> ...
    HqBuildState          graph/hq_build.build_build_graph         generate_all -> filter_all -> write (was BuildState)
    HqQuestion, TableQuestions                                      what the question LLM must return

Each node sets the fields named after it; `trace` holds node -> {"seconds": float, ...}.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.schemas.resources import ChromaConfig
from kosis_rag.shared.fields import TableFields
from kosis_rag_build.hq.constants import FILTER_FRAC, MIN_KEEP

Trace = dict[str, dict[str, Any]]


# --- table registry, 시군구 dictionary, Chroma table index, question index ---


def index_version_now() -> str:
    """When a load happens, as the old CLIs stamped it: local time YYYYMMDDHHMM (T2-41 S6b: the states' default)."""
    return datetime.now().strftime("%Y%m%d%H%M")

class TablesBuildState(BaseModel):
    """data/tables.jsonl from the provided data (tables.py). data_dir / out are the input."""

    data_dir: Path
    out: Path
    sources: list[dict[str, Any]] = Field(default_factory=list)  # read_parquet: one tables.read_table(...) per row
    records: list[TableRecord] = Field(default_factory=list)  # build_records: one per tables.csv row, in file order
    count: int = 0  # write_jsonl: lines written
    trace: Trace = Field(default_factory=dict)


class SigunguBuildState(BaseModel):
    """data/sigungu.json (시군구 -> 시도) from one 읍면동 table (sigungu.py). parquet / out are the input."""

    parquet: Path
    out: Path
    rows: list[dict[str, Any]] = Field(default_factory=list)  # read_parquet: distinct {dim1_code, dim1_label}
    mapping: dict[str, list[str]] = Field(default_factory=dict)  # build_sigungu
    trace: Trace = Field(default_factory=dict)


class IndexBuildState(BaseModel):
    """One table index build (chroma.py). tables ... export are the input (the CLI fills them from its arguments)."""

    tables: Path
    model: Literal["kure", "bge-m3-ft"] = "kure"
    ckpt: str | None = None  # bge-m3-ft: checkpoint .pt or run folder
    text_format: str | None = None  # bge-m3-ft: kosis_rag.shared.tabletext format (None: its DOC_FORMAT)
    collection: str | None = None  # None: chroma.collection_name / FT_COLLECTION_PREFIX + index_version
    device: str | None = None
    export: Path | None = None  # write the build to this .npz instead of Chroma
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)  # server for load
    index_version: str = Field(default_factory=index_version_now)  # YYYYMMDDHHMM of this load
    data_sha256: str = ""  # read_records
    name: str = ""  # read_records: the collection name
    ids: list[str] = Field(default_factory=list)  # read_records: table keys
    metas: list[dict[str, Any]] = Field(default_factory=list)  # read_records: chroma.to_metadata
    records: list[TableRecord] = Field(default_factory=list)  # read_records
    docs: list[str] = Field(default_factory=list)  # build_docs
    fmt: str | None = None  # build_docs: the text format embedded (bge-m3-ft only)
    vectors: list[list[float]] = Field(default_factory=list)  # embed
    record: dict[str, Any] = Field(default_factory=dict)  # embed: chroma.build_record (collection metadata)
    count: int = 0  # load: documents in the collection afterwards (write_build: documents written)
    trace: Trace = Field(default_factory=dict)


class QuestionsIndexState(BaseModel):
    """One question index load (hq/questions_index.py). build ... dry_run are the input."""

    build: str  # build id (with -desc for the descriptions)
    ckpt: str | None = None  # fine-tuned bge-m3 .pt (EMBED_CKPT)
    device: str | None = None
    collection: str | None = None  # None: kosis_rag.shared.hqnames.collection_name(build, ckpt id)
    hq_root: Path | None = None  # data/hq (None: kosis_rag.adapters.files.HQ_DIR)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    dry_run: bool = False  # read_questions only: data_sha256, name and count, no embedding, no Chroma
    index_version: str = Field(default_factory=index_version_now)  # YYYYMMDDHHMM of this load
    rows: list[dict[str, Any]] = Field(default_factory=list)  # read_questions
    texts: list[str] = Field(default_factory=list)  # read_questions: norm_query(text)
    data_sha256: str = ""  # read_questions: questions_index.data_sha256(rows)
    ckpt_id: str = ""  # read_questions (dry run) / embed: the embedder weights id
    name: str = ""  # read_questions (dry run) / embed: the collection name
    vectors: list[list[float]] = Field(default_factory=list)  # embed
    record: dict[str, Any] = Field(default_factory=dict)  # embed: collection metadata
    count: int = 0  # load
    trace: Trace = Field(default_factory=dict)


# --- question generation (T2-39) ---

QType = Literal["concept", "condition", "distinct", "keyword"]
TYPES: tuple[str, ...] = ("concept", "condition", "distinct", "keyword")
MAX_QUESTIONS = 12  # the prompt asks for 6; more than this is a malformed reply


class HqQuestion(BaseModel):
    """One hypothetical question: how a person might ask for the table."""

    type: QType
    text: str = Field(min_length=2, max_length=200)


class TableQuestions(BaseModel):
    """The LLM reply for one table (prompts/hq-v1.md). key is filled in by the code, not taken from the reply."""

    key: str
    description: str = Field(min_length=2, max_length=400)  # one sentence: what the table shows, broken down by what
    questions: list[HqQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)


class HqState(BaseModel):
    """One table through the generation graph (graph.hq_build.build_table_graph); the nodes run in a line and each
    sets the fields named after it below."""

    key: str
    record: TableRecord | None = None  # load
    fields: TableFields | None = None  # load
    doc: str = ""  # load: the table's n2 embedding text (kosis_rag.shared.tabletext.build_doc)
    messages: list[dict[str, str]] = Field(default_factory=list)  # prompt: system + user; validate adds re-ask turns
    prompt_hash: str = ""  # prompt: hash of the fixed template + version (cache folder); the table's own text is
    # keyed separately (nodes.user_sha, in the cache file name)
    raw: str | None = None  # generate: the last reply text
    responses: list[dict[str, Any]] = Field(default_factory=list)  # generate: every reply {raw, usage, seconds, cached}
    attempts: int = 0  # generate: replies so far (1 = first answer, 2 = after the one re-ask)
    description: str = ""  # validate
    questions: list[HqQuestion] = Field(default_factory=list)  # validate, then rule_check keeps the passing ones
    positions: list[int] = Field(default_factory=list)  # rule_check: each kept question's place in the reply (its id)
    dropped: dict[str, str] = Field(default_factory=dict)  # rule_check: question text -> reason
    scores: list[float] = Field(default_factory=list)  # score: one per kept question (empty with no scorer)
    error: str | None = None  # generate (API failure) or validate (reply still invalid after the re-ask)
    trace: Trace = Field(default_factory=dict)  # node -> {"seconds": float, ...}


class HqBuildState(BaseModel):
    """The whole build (graph.hq_build.build_build_graph; was hq.graph.BuildState). keys/tables_sha/frac/min_keep/
    limit are the input."""

    keys: list[str]
    tables_sha: str
    model: str
    frac: float = FILTER_FRAC
    min_keep: int = MIN_KEEP
    limit: int | None = None
    results: list[HqState] = Field(default_factory=list)  # generate_all: one state per table, in key order
    items: list[dict] = Field(default_factory=list)  # filter_all: every question that passed the rules
    kept: list[dict] = Field(default_factory=list)  # filter_all: what goes into questions.jsonl
    score_dropped: list[dict] = Field(default_factory=list)  # filter_all
    build_id: str = ""  # write
    out_dir: str = ""  # write
    totals: dict[str, Any] = Field(default_factory=dict)  # write: tables, questions, tokens, calls, seconds
    trace: Trace = Field(default_factory=dict)
