"""Run a Phase 1 module from this repo, with KOSIS_WORKSPACE defaulting to this repo (docs/ISSUES.md P2-07).

    python scripts/phase1.py run_eval --stage x --golden eval/golden/golden.jsonl --pipeline stat ...
    python scripts/phase1.py grade_run --run eval/results/<run>
    python scripts/phase1.py kosis_chat.cli --table ...      # a dotted name is used as is

A bare name means eval.<name>. Same as `cd vendor/phase1`, `set KOSIS_WORKSPACE=..\\..`, `python -m eval.<name> ...`.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "vendor" / "phase1"


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    os.environ.setdefault("KOSIS_WORKSPACE", str(ROOT))
    sys.path[:0] = [str(PHASE1), str(PHASE1 / "src")]
    module = sys.argv[1] if "." in sys.argv[1] else f"eval.{sys.argv[1]}"
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
