"""KOSIS_WORKSPACE (docs/ISSUES.md P2-07): importing kosis_rag leaves the environment alone (T2-41 S1, P2-D5); the
Phase 1 runner script still defaults it to this repo, so Phase 1 reads data/, eval/golden and .env here."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path, **env: str) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if k != "KOSIS_WORKSPACE"} | env
    return subprocess.run([sys.executable, *args], cwd=cwd, env=clean, capture_output=True, text=True)


def test_importing_kosis_rag_does_not_change_the_environment():
    code = "import os, kosis_rag; assert 'KOSIS_WORKSPACE' not in os.environ; print('clean')"
    path = os.pathsep.join([str(ROOT / "src"), str(ROOT / "vendor" / "phase1" / "src")])
    out = _run(["-c", code], ROOT, PYTHONPATH=path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "clean"
    kept = _run(["-c", "import os, kosis_rag; print(os.environ['KOSIS_WORKSPACE'])"], ROOT, PYTHONPATH=path,
                KOSIS_WORKSPACE=str(ROOT / "elsewhere"))
    assert Path(kept.stdout.strip()) == ROOT / "elsewhere"  # a value the caller set is left as it is


def test_phase1_runner_works_from_any_folder_with_this_repo_as_workspace(tmp_path):
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "__init__.py").write_text("")
    (probe / "show.py").write_text("from kosis_ai import paths\nprint(paths.data_dir())\n")
    out = _run([str(ROOT / "scripts" / "phase1.py"), "probe.show"], ROOT / "frontend", PYTHONPATH=str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()) == ROOT / "data"
    help_ = _run([str(ROOT / "scripts" / "phase1.py"), "run_eval", "--help"], ROOT / "frontend")
    assert help_.returncode == 0 and "--stage" in help_.stdout  # a bare name runs eval.<name> with its arguments
