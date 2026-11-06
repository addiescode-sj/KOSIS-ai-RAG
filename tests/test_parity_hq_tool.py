"""T2-41 S6a: scripts/parity_hq.py replay runs on the moved build code (kosis_rag_build.graph.hq_build, hq.nodes,
hq.prompt): a build made with a fake LLM replays from its cache alone, identical, with no call; a cache miss stops
before anything is written (exit 3); an unsafe --out-dir is refused (exit 2). No model, no API, all under tmp_path.
The filtered (score) replay needs the stage 1 checkpoint and runs on the PC only (plans/T2-41.md S6a Done ③).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from kosis_rag_build.graph.hq_build import run_build  # noqa: E402
from kosis_rag_build.hq import nodes as hq_nodes  # noqa: E402
from tests.test_hq_graph import ONE, TWO, FakeLLM, _reply  # noqa: E402

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[1]
MODEL = "fake-mini"


def _tool():
    spec = importlib.util.spec_from_file_location("parity_hq", ROOT / "scripts" / "parity_hq.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(tmp_path) -> tuple[Path, Path, Path]:
    """tables.jsonl, the data/hq-like root (cache + one build folder) and the build folder."""
    tables = tmp_path / "tables.jsonl"
    tables.write_text("".join(r.model_dump_json() + "\n" for r in (ONE, TWO)), encoding="utf-8")
    root = tmp_path / "hq"
    two = ["혼인 안 한 사람이 얼마나 되나", "결혼한 사람 비율 변화", "2020년 서울 미혼 인구",
           "대구 미혼 남성", "2020년 인구총조사 혼인상태 시도별", "시도 혼인상태 인구"]
    state = run_build([ONE, TWO], FakeLLM("not json", _reply(), _reply(two)), model=MODEL,
                      tables_sha256=hq_nodes.tables_sha(tables), cache_dir=root / "cache", out_root=root,
                      concurrency=1, log=lambda s: None)
    return tables, root, Path(state.out_dir)


def test_replay_of_a_cached_build_is_identical_and_calls_nothing(tmp_path, capsys):
    tables, root, build = _baseline(tmp_path)
    out = tmp_path / "replay"
    code = _tool().main(["replay", str(build), "--out-dir", str(out), "--tables", str(tables), "--concurrency", "1"])
    assert code == 0 and "identical" in capsys.readouterr().out
    for name in ("questions.jsonl", "descriptions.jsonl", "failed.jsonl"):
        assert (out / build.name / name).read_bytes() == (build / name).read_bytes()
    replayed = json.loads((out / build.name / "build.json").read_text(encoding="utf-8"))
    assert replayed["totals"]["calls"] == 0 and replayed["totals"]["cached"] == 3
    assert (out / f"{build.name}.sha256").is_file()


def test_a_cache_miss_stops_the_replay_before_writing(tmp_path, capsys):
    tables, root, build = _baseline(tmp_path)
    for path in (root / "cache").rglob("101_DT_TWO-*.json"):
        path.unlink()
    out = tmp_path / "replay"
    code = _tool().main(["replay", str(build), "--out-dir", str(out), "--tables", str(tables), "--concurrency", "1"])
    assert code == 3 and not (out / build.name).exists()
    capsys.readouterr()


def test_an_unsafe_out_dir_is_refused(tmp_path, capsys):
    tables, root, build = _baseline(tmp_path)
    assert _tool().main(["replay", str(build), "--out-dir", str(build / "x"), "--tables", str(tables)]) == 2
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f").write_text("x")
    assert _tool().main(["replay", str(build), "--out-dir", str(tmp_path / "full"), "--tables", str(tables)]) == 2
    capsys.readouterr()
