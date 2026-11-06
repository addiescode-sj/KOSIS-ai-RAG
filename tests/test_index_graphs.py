"""T2-41 S6a: the four offline index graphs (kosis_rag_build.graph.index_build) write what the old CLIs wrote.

The old CLI functions are frozen below (bodies copied from the S5 tree: kosis_rag.metadata.main / build_records,
kosis_rag.shared.vocab.build_sigungu / main, kosis_rag.index.main (the --export path) and kosis_rag.hq.index.main),
with the model and the Chroma client passed in instead of loaded, and the time stamp passed in instead of read. Each
graph runs on fake Parquet files, a fake embedder and a fake Chroma client, and its output is compared byte for byte
(or value for value, for what goes to Chroma) with the frozen function's. Kept for the moving step only.
Also: the CLI parses its arguments and the shell environment into the settings objects (no .env, no network).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("langgraph")

from kosis_rag.adapters import files  # noqa: E402
from kosis_rag.adapters import models  # noqa: E402
from kosis_rag.shared.hqnames import HQ_DOC_FORMAT, collection_name  # noqa: E402
from kosis_rag.schemas.catalog import TableRecord  # noqa: E402
from kosis_rag.schemas.common import TableInfo  # noqa: E402
from kosis_rag.shared import tabletext as text  # noqa: E402
from kosis_rag.shared.units import meta_unit  # noqa: E402
from kosis_rag.shared.vocab import SIDO_LABELS  # noqa: E402
from kosis_rag_build import chroma, cli, sigungu, tables  # noqa: E402
from kosis_rag_build.graph.index_build import (build_chroma_graph, build_questions_graph,  # noqa: E402
                                               build_sigungu_graph, build_tables_graph)
from kosis_rag_build.schemas import (IndexBuildState, QuestionsIndexState, SigunguBuildState,  # noqa: E402
                                     TablesBuildState)

pytestmark = pytest.mark.offline

INDEX_VERSION = "202611060000"


# --- fake provided data ---

def _row(key, name):
    return {"key": key, "org_id": "101", "tbl_id": key.split("_", 1)[1], "survey_name": "인구총조사",
            "organization": "국가데이터처", "table_name": name, "period": "5년 (2015 ~ 2025)", "source": "api",
            "coverage": "A_정상", "missing_periods": "", "rows": "4", "numeric_rows": "4", "symbol_rows": "0",
            "parquet": f"values/{key}.parquet", "url": f"http://kosis.kr/{key}", "menu_path": "인구 > 가구"}


ROWS = [_row("101_DT_A", "1인 가구"), _row("101_DT_B", "시도/연령/출생아수별 부인"), _row("101_DT_C", "값 없는 표")]
REGIONS = [("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("11010", "종로구"),
           ("21020", "중구"), ("11020", "중구"), ("22010", "중구"), ("1101053", "사직동")]


def _data_dir(root: Path) -> Path:
    for sub in ("values", "reference", "table_metadata"):
        (root / sub).mkdir(parents=True)
    with (root / "tables.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)
    catalog = [{"org_id": "101", "tbl_id": "DT_A", "topic": "인구",
                "catalog_periods": [{"frequency_label": "5년", "start": "2015", "end": "2025"}],
                "paths": [{"path": "인구 > 가구"}, {"path": "인구 > 인구총조사 > 가구"}]}]
    (root / "reference" / "population_table_catalog.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in catalog), encoding="utf-8")
    (root / "table_metadata" / "101_DT_A.json").write_text(
        json.dumps({"tblEngNm": "One-person households", "unitNmKor": "가구", "periodStr": "F",
                    "containPeriod": " 5년 2015 ~ 2025"}), encoding="utf-8")
    for key, item, dims in (("101_DT_A", "1인 가구 (가구)", ["행정구역(읍면동)별"]),
                            ("101_DT_B", "부인 (명)", ["행정구역별", "연령별"])):
        cells = [(p, code, label) for p in ("2015", "2020") for code, label in REGIONS]
        frame = {"prd_se": "Y", "period": [p for p, _, _ in cells], "item_name": item,
                 "dim1_name": dims[0], "dim1_code": [c for _, c, _ in cells], "dim1_label": [r for _, _, r in cells]}
        if len(dims) > 1:
            frame |= {"dim2_name": dims[1], "dim2_code": "A", "dim2_label": "15~19세"}
        else:
            frame |= {"dim2_name": None, "dim2_code": None, "dim2_label": None}
        pd.DataFrame(frame).to_parquet(root / "values" / f"{key}.parquet")
    # the 읍면동 table sigungu.json is read from
    pd.DataFrame({"dim1_code": [c for c, _ in REGIONS] * 2, "dim1_label": [r for _, r in REGIONS] * 2,
                  "period": "2020"}).to_parquet(root / sigungu.SIGUNGU_SOURCE)
    return root


# --- frozen old CLI functions (S5 tree), model / client / time passed in ---

def old_metadata_build_records(data_dir: Path):
    """kosis_rag.metadata.build_records (S5), verbatim but for the module prefix of its helpers."""
    with (data_dir / tables.CATALOG).open(encoding="utf-8") as f:
        catalog = {f"{c['org_id']}_{c['tbl_id']}": c for c in map(json.loads, f)}
    with (data_dir / "tables.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        info = TableInfo(**row)
        meta_path = data_dir / "table_metadata" / f"{info.key}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        cat = catalog.get(info.key)
        parquet = data_dir / info.parquet
        values = tables._values(parquet) if parquet.exists() else {}
        periods = values.get("periods", [])
        years = [y for y in map(tables._year, periods) if y is not None]
        record = TableRecord(
            info=info,
            table_name_en=(meta or {}).get("tblEngNm") or None,
            topic=(cat or {}).get("topic"),
            menu_paths=list(dict.fromkeys([info.menu_path, *(p["path"] for p in (cat or {}).get("paths", []))])),
            unit=meta_unit(info.key, data_dir),
            cycle=[c for c in ((meta or {}).get("periodStr") or "").split("#") if c],
            contain_period=((meta or {}).get("containPeriod") or "").strip(),
            start=min(years, default=None),
            end=max(years, default=None),
            **values,
        )
        record.issues = tables._issues(record, cat, meta)
        record.search_text = tables._search_text(record)
        yield record


def old_metadata_main(data_dir: Path, out: Path) -> int:
    """kosis_rag.metadata.main (S5) without argparse."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in old_metadata_build_records(data_dir):
            f.write(record.model_dump_json() + "\n")
    return 0


def old_build_sigungu(parquet: Path) -> dict[str, list[str]]:
    """kosis_rag.shared.vocab.build_sigungu (S5), verbatim."""
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=["dim1_code", "dim1_label"]).group_by(["dim1_code", "dim1_label"]) \
        .aggregate([]).to_pylist()
    sido = {r["dim1_code"]: SIDO_LABELS[r["dim1_label"]] for r in rows
            if len(r["dim1_code"] or "") == 2 and r["dim1_label"] in SIDO_LABELS}
    out: dict[str, set[str]] = {}
    for r in rows:
        code, label = r["dim1_code"] or "", r["dim1_label"]
        if len(code) == 5 and code[:2] in sido:
            out.setdefault(label, set()).add(sido[code[:2]])
    return {label: sorted(names) for label, names in sorted(out.items())}


def old_vocab_main(data_dir: Path, out: Path) -> int:
    """kosis_rag.shared.vocab.main (S5) without argparse."""
    mapping = old_build_sigungu(data_dir / "values/101_DT_1IN1502.parquet")
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    return 0


def old_index_export(tables_path: Path, export: Path, *, ft: bool, embedder, index_version: str,
                     text_format: str | None = None, collection: str | None = None) -> int:
    """kosis_rag.index.main (S5), the --export path; embedder = (embed, count_tokens, max_tokens, info) as the old
    main got them from kure_embedder / model.load_embedder."""
    data_sha256 = hashlib.sha256(tables_path.read_bytes()).hexdigest()
    name = collection or (chroma.FT_COLLECTION_PREFIX + index_version if ft else chroma.collection_name(data_sha256))
    with tables_path.open(encoding="utf-8") as f:
        records = [TableRecord.model_validate_json(line) for line in f]
    ids = [r.info.key for r in records]
    metas = [chroma.to_metadata(r) for r in records]
    if ft:
        fmt = text_format or text.DOC_FORMAT
        docs = [text.build_doc(r, fmt)["embed_text"] for r in records]
        embed, count_tokens, max_tokens = embedder.embed, embedder.count_tokens, embedder.max_tokens
        info = {"embed_model": embedder.info["embed_model"], "doc_format": fmt, "embed_ckpt": embedder.info["embed_ckpt"]}
    else:
        docs = [chroma.to_document(r) for r in records]
        embed, count_tokens, max_tokens = embedder.embed, embedder.count_tokens, embedder.max_tokens
        info = {}
    tokens = [count_tokens(d) for d in docs]
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    vectors: list = [None] * len(docs)
    for chunk in [order[i:i + 64] for i in range(0, len(order), 64)]:
        for i, v in zip(chunk, embed([docs[i] for i in chunk])):
            vectors[i] = v
    record = chroma.build_record(index_version, data_sha256, len(docs), len(vectors[0]), max(tokens),
                                 sum(t > max_tokens for t in tokens), **info)
    chroma.write_build(export, name=name, ids=ids, docs=docs, metas=metas, vectors=vectors, record=record)
    return 0


def old_hq_index_main(build: str, root: Path, embedder, client, index_version: str) -> int:
    """kosis_rag.hq.index.main (S5) with the embedder, Chroma client, build root and time stamp passed in."""
    rows = files.read_questions(build, root)
    texts = [text.norm_query(r["text"]) for r in rows]
    vectors = embedder.embed_documents(texts)
    name = collection_name(build, embedder.ckpt)
    digest = hashlib.sha256("\n".join(f"{r['id']}\t{r['text']}" for r in rows).encode("utf-8")).hexdigest()
    record = {"index_version": index_version, "doc_format": HQ_DOC_FORMAT, "hq_build": build,
              "embed_model": embedder.name, "embed_ckpt": embedder.ckpt, "data_sha256": digest,
              "dim": len(vectors[0]), "normalized": True, "hnsw:space": "cosine", "n_docs": len(rows),
              "n_tables": len({r["key"] for r in rows})}
    count = chroma.load(client, [r["id"] for r in rows], texts, [{"key": r["key"], "type": r["type"]} for r in rows],
                        vectors, name, record)
    return 0 if count == len(rows) else 1


# --- fakes ---

def _vector(t: str) -> list[float]:
    h = hashlib.sha256(t.encode("utf-8")).digest()
    return [b / 255 for b in h[:4]]


def _embed(texts: list[str]) -> list[list[float]]:
    return [_vector(t) for t in texts]


KURE = chroma.DocEmbedder(_embed, len, 1024, {})
FT = chroma.DocEmbedder(_embed, len, 1024, {"embed_model": "BAAI/bge-m3", "doc_format": "n2-L60",
                                            "embed_ckpt": "fake.pt@0123456789ab"})


class FakeQuestionEmbedder:
    name = "BAAI/bge-m3"
    ckpt = "unseen-e1.pt@abcdef012345"

    def embed_documents(self, texts):
        return _embed(texts)


class FakeCollection:
    def __init__(self, metadata):
        self.metadata, self.rows = metadata, {}

    def modify(self, metadata):
        self.metadata = metadata

    def upsert(self, ids, documents, metadatas, embeddings):
        self.rows.update({i: (d, m, e) for i, d, m, e in zip(ids, documents, metadatas, embeddings)})

    def get(self, include):
        return {"ids": list(self.rows)}

    def delete(self, ids):
        for i in ids:
            del self.rows[i]

    def count(self):
        return len(self.rows)


class FakeClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name, embedding_function, metadata):
        return self.collections.setdefault(name, FakeCollection(metadata))

    def get_max_batch_size(self):
        return 2


# --- the graphs against the frozen functions ---

def _tables(tmp_path) -> tuple[Path, Path]:
    data = _data_dir(tmp_path / "data")
    out = tmp_path / "new" / "tables.jsonl"
    state = TablesBuildState.model_validate(build_tables_graph().invoke(TablesBuildState(data_dir=data, out=out)))
    assert state.count == len(ROWS) and list(state.trace) == ["read_parquet", "build_records", "write_jsonl"]
    return data, out


def test_tables_graph_writes_the_old_tables_jsonl(tmp_path):
    data, out = _tables(tmp_path)
    old_metadata_main(data, tmp_path / "old" / "tables.jsonl")
    assert out.read_bytes() == (tmp_path / "old" / "tables.jsonl").read_bytes()
    assert [TableRecord.model_validate_json(x).info.key for x in out.read_text(encoding="utf-8").splitlines()] == [
        r["key"] for r in ROWS]
    assert [r.model_dump_json() for r in tables.build_records(data)] == out.read_text(encoding="utf-8").splitlines()


def test_sigungu_graph_writes_the_old_sigungu_json(tmp_path):
    data = _data_dir(tmp_path / "data")
    out, old = tmp_path / "sigungu.json", tmp_path / "old.json"
    state = SigunguBuildState.model_validate(build_sigungu_graph().invoke(
        SigunguBuildState(parquet=data / sigungu.SIGUNGU_SOURCE, out=out)))
    old_vocab_main(data, old)
    assert out.read_bytes() == old.read_bytes()
    assert state.mapping == {"종로구": ["서울특별시"], "중구": ["대구광역시", "부산광역시", "서울특별시"]}
    assert sigungu.build_sigungu(data / sigungu.SIGUNGU_SOURCE) == state.mapping


@pytest.mark.parametrize("model", ["kure", "bge-m3-ft"])
def test_chroma_graph_exports_the_old_build(tmp_path, model):
    _, tables_path = _tables(tmp_path)
    ft = model == "bge-m3-ft"
    new, old = tmp_path / "new.npz", tmp_path / "old.npz"
    state = IndexBuildState(tables=tables_path, model=model, export=new, index_version=INDEX_VERSION)
    done = IndexBuildState.model_validate(build_chroma_graph(embedder=FT if ft else KURE, log=lambda s: None)
                                          .invoke(state))
    old_index_export(tables_path, old, ft=ft, embedder=FT if ft else KURE, index_version=INDEX_VERSION)
    a, b = chroma.read_build(new), chroma.read_build(old)
    assert a == b  # name, ids, docs, metas, record (data_sha256 included) and vectors
    assert list(done.trace) == ["read_records", "build_docs", "embed", "write_build"] and done.count == len(ROWS)
    assert a["record"]["data_sha256"] == hashlib.sha256(tables_path.read_bytes()).hexdigest()


def test_chroma_graph_loads_through_the_given_client(tmp_path):
    _, tables_path = _tables(tmp_path)
    client = FakeClient()
    state = IndexBuildState.model_validate(build_chroma_graph(embedder=KURE, client=client, log=lambda s: None)
                                           .invoke(IndexBuildState(tables=tables_path, index_version=INDEX_VERSION)))
    collection = client.collections[state.name]
    assert state.count == len(ROWS) and sorted(collection.rows) == sorted(r["key"] for r in ROWS)
    assert list(state.trace) == ["read_records", "build_docs", "embed", "load"]
    assert collection.metadata == state.record and state.name.startswith(chroma.COLLECTION_PREFIX)


def _hq_build(root: Path, build: str) -> None:
    folder = root / build
    folder.mkdir(parents=True)
    rows = [{"id": "101_DT_A#q0", "key": "101_DT_A", "type": "concept", "text": "혼자 사는 사람 수"},
            {"id": "101_DT_A#q2", "key": "101_DT_A", "type": "condition", "text": "2020년 서울 1인 가구"},
            {"id": "101_DT_B#q1", "key": "101_DT_B", "type": "keyword", "text": "출생아 수 부인"}]
    (folder / "questions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                            encoding="utf-8")
    (folder / "descriptions.jsonl").write_text(
        json.dumps({"key": "101_DT_A", "description": "1인 가구를 시도별로"}, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.mark.parametrize("build", ["hq-cb0b3c5b-a3c88540-fake-f30-k2", "hq-cb0b3c5b-a3c88540-fake-f30-k2-desc"])
def test_questions_graph_loads_what_the_old_cli_loaded(tmp_path, build):
    root = tmp_path / "hq"
    _hq_build(root, build.removesuffix("-desc"))
    new, old = FakeClient(), FakeClient()
    state = QuestionsIndexState.model_validate(build_questions_graph(embedder=FakeQuestionEmbedder(), client=new).invoke(
        QuestionsIndexState(build=build, hq_root=root, index_version=INDEX_VERSION)))
    assert old_hq_index_main(build, root, FakeQuestionEmbedder(), old, INDEX_VERSION) == 0
    assert set(new.collections) == set(old.collections) == {state.name}
    a, b = new.collections[state.name], old.collections[state.name]
    assert a.rows == b.rows and a.metadata == b.metadata
    assert list(state.trace) == ["read_questions", "embed", "load"] and state.count == len(state.rows)


def test_questions_dry_run_reads_only(tmp_path):
    root, build = tmp_path / "hq", "hq-cb0b3c5b-a3c88540-fake-f30-k2"
    _hq_build(root, build)
    ckpt = tmp_path / "unseen-e1.pt"
    ckpt.write_bytes(b"weights")
    client = FakeClient()

    class NoEmbedder:
        def embed_documents(self, texts):
            raise AssertionError("a dry run embeds nothing")

    state = QuestionsIndexState.model_validate(build_questions_graph(embedder=NoEmbedder(), client=client).invoke(
        QuestionsIndexState(build=build, hq_root=root, ckpt=str(ckpt), dry_run=True)))
    assert client.collections == {} and list(state.trace) == ["read_questions"]
    ckpt_id = models.ckpt_id(models.resolve_ckpt(ckpt))
    assert state.ckpt_id == ckpt_id and state.name == collection_name(build, ckpt_id)
    rows = files.read_questions(build, root)
    assert state.data_sha256 == hashlib.sha256(
        "\n".join(f"{r['id']}\t{r['text']}" for r in rows).encode("utf-8")).hexdigest()


# --- the CLI: arguments and the shell environment -> settings objects ---

ENV = {"CHROMA_HOST": "chroma.local", "CHROMA_PORT": "9000", "CHROMA_COLLECTION": "kosis_kure__x__1",
       "EMBED_CKPT": "data/models/e.pt", "RERANK_CKPT": "data/models/r.pt", "TEXT_FORMAT": "n2-L60",
       "OPENAI_API_KEY": "sk-test"}


def test_cli_settings_come_from_arguments_then_the_environment():
    args = cli.parser().parse_args(["chroma", "--model", "bge-m3-ft"])
    spec = cli.index_spec(args, ENV)
    assert (spec.chroma.host, spec.chroma.port, spec.chroma.collection) == ("chroma.local", 9000, "kosis_kure__x__1")
    assert (spec.embed_ckpt, spec.text_format) == ("data/models/e.pt", "n2-L60")
    args = cli.parser().parse_args(["chroma", "--ckpt", "a.pt", "--text-format", "n3-L60", "--chroma-host", "h",
                                    "--chroma-port", "1"])
    spec = cli.index_spec(args, ENV)
    assert (spec.embed_ckpt, spec.text_format, spec.chroma.host, spec.chroma.port) == ("a.pt", "n3-L60", "h", 1)
    spec = cli.index_spec(cli.parser().parse_args(["chroma"]), {})
    assert (spec.chroma.host, spec.chroma.port, spec.chroma.collection, spec.embed_ckpt) == ("localhost", 8001, None, None)


def test_cli_hq_build_llm_config():
    from kosis_rag_build.hq import constants

    args = cli.parser().parse_args(["hq-build", "--model", "gpt-4o-mini", "--limit", "20"])
    llm = cli.hq_llm_config(args, ENV)
    assert (llm.provider, llm.model, llm.api_key.get_secret_value()) == ("openai", "gpt-4o-mini", "sk-test")
    assert (llm.temperature, llm.max_tokens, llm.seed) == (constants.TEMPERATURE, constants.MAX_TOKENS, constants.SEED)
    assert cli.hq_llm_config(args, {}).api_key is None
    assert cli.index_spec(args, ENV).rerank_ckpt == "data/models/r.pt"
    assert (args.frac, args.min_keep, args.text_format) == (constants.FILTER_FRAC, constants.MIN_KEEP, text.DOC_FORMAT)


def test_cli_hq_build_runs_the_graph_without_calling_the_llm(tmp_path, monkeypatch, capsys):
    from kosis_rag.llm.openai import OpenAIClient
    from kosis_rag_build.graph import hq_build
    from kosis_rag_build.schemas import HqBuildState

    _, tables_path = _tables(tmp_path)
    seen = {}

    def fake_run_build(records, client, **kw):
        seen.update(kw, client=client, n=len(records))
        return HqBuildState(keys=[], tables_sha=kw["tables_sha256"], model=kw["model"], build_id="hq-x", out_dir="o",
                            totals={k: 0 for k in ("tables", "failed", "generated", "rule_dropped", "score_dropped",
                                                   "kept", "replies", "calls", "cached", "api_seconds", "input_tokens",
                                                   "output_tokens", "new_input_tokens", "new_output_tokens")})

    monkeypatch.setattr(hq_build, "run_build", fake_run_build)
    monkeypatch.setattr(OpenAIClient, "generate", lambda *a, **k: pytest.fail("no LLM call in a test"))
    assert cli.main(["hq-build", "--tables", str(tables_path), "--no-score", "--limit", "2"], env={}) == 0
    assert isinstance(seen["client"], OpenAIClient) and seen["client"].config.api_key is None
    assert seen["scorer"] is None and seen["limit"] == 2 and seen["n"] == len(ROWS)
    assert seen["tables_sha256"] == hashlib.sha256(tables_path.read_bytes()).hexdigest()
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY is empty" in out and "tokens this run" in out and "cost" in out
    with pytest.raises(SystemExit):  # a filtered build needs the stage 1 checkpoint
        cli.main(["hq-build", "--tables", str(tables_path)], env={})


def test_cli_commands_run_the_graphs(tmp_path, monkeypatch, capsys):
    data = _data_dir(tmp_path / "data")
    out = tmp_path / "t.jsonl"
    assert cli.main(["tables", "--data-dir", str(data), "--out", str(out)], env={}) == 0
    assert cli.main(["sigungu", "--data-dir", str(data), "--out", str(tmp_path / "s.json")], env={}) == 0
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["종로구"] == ["서울특별시"]
    seen = {}

    def fake_embedder(state, log=print):
        seen["state"] = state
        return FT

    monkeypatch.setattr(chroma, "load_doc_embedder", fake_embedder)
    npz = tmp_path / "b.npz"
    assert cli.main(["chroma", "--tables", str(out), "--model", "bge-m3-ft", "--export", str(npz)], env=ENV) == 0
    assert seen["state"].ckpt == "data/models/e.pt" and seen["state"].fmt == "n2-L60"
    assert chroma.read_build(npz)["name"].startswith(chroma.FT_COLLECTION_PREFIX)
    root, build = tmp_path / "hq", "hq-cb0b3c5b-a3c88540-fake-f30-k2"
    _hq_build(root, build)
    ckpt = tmp_path / "unseen-e1.pt"
    ckpt.write_bytes(b"weights")
    capsys.readouterr()
    assert cli.main(["hq-index", "--build", build, "--hq-root", str(root), "--dry-run"],
                    env={"EMBED_CKPT": str(ckpt)}) == 0
    line = capsys.readouterr().out
    assert "dry run" in line and collection_name(build, models.ckpt_id(ckpt.resolve())) in line
