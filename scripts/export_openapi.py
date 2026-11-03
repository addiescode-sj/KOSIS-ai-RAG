"""Write the API's OpenAPI document to docs/openapi.json without starting a server (T2-37).

Usage: PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src uv run python scripts/export_openapi.py
The same document is served live at /openapi.json, with Swagger UI at /docs and ReDoc at /redoc.
"""

import json
from pathlib import Path

from backend.app.main import app

out = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
out.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"{out} ({len(app.openapi()['paths'])} paths)")
