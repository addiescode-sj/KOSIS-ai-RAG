"""Run the offline build CLI with the shell environment as the settings fallback (T2-41 S6a, 미결 3).

    python scripts/kosis_build.py {tables|sigungu|chroma|text|hq-build|hq-index} ...

Same commands and arguments as `python -m kosis_rag_build.cli` (see its docstring). This script is the one place
that reads the environment for the build: CHROMA_HOST CHROMA_PORT CHROMA_COLLECTION EMBED_CKPT RERANK_CKPT
TEXT_FORMAT OPENAI_API_KEY from the shell (`set X=...` first). No .env is read.
(Windows cmd: `set PYTHONPATH=.;src;vendor/phase1;vendor/phase1/src`.)
"""

import os
import sys

from kosis_rag_build.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], env=os.environ))
