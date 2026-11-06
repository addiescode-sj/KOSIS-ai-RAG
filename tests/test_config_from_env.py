"""T2-41 S4: backend/app/config.py reads the shell environment (never .env) into SearchOptions / IndexSpec / LLMConfig
with the keys and defaults the deleted SearchSettings.from_env used, plus the guide LLM mapping (plans/T2-41.md S4).

The old defaults are written out below (SearchSettings at T2-41 S3, before it was deleted), so a drift between the
old and the new defaults fails here instead of changing L1 (parity gate 1).
"""

from pathlib import Path

import pytest

from backend.app import config
from backend.app.config import SearchConfig, search_config_from_env
from kosis_rag.adapters import files
from kosis_rag.llm.config import require_key
from kosis_rag.schemas.resources import IndexSpec, SearchOptions

pytestmark = pytest.mark.offline

REPO = Path(__file__).resolve().parents[1]
ENV_KEYS = ("CHROMA_HOST", "CHROMA_PORT", "CHROMA_COLLECTION", "TEXT_FORMAT", "EMBED_CKPT", "RERANK_CKPT", "RERANK_MODE",
            "STAGE2", "COND_FILTER", "ANSWER_K", "HQ_BUILD", "HQ_WEIGHT", "HQ_BM25", "HCX_API_KEY", "HCX_THINKING",
            "OPENAI_API_KEY", "CLOVASTUDIO_API_KEY", "GUIDE_MODEL", "GUIDE_TEMPERATURE", "GUIDE_MAX_TOKENS",
            "GUIDE_TIMEOUT")
# SearchSettings() / SearchSettings.from_env() with an empty environment, field by field (T2-41 S3 pipeline.py)
OLD_DEFAULTS = {
    "tables": REPO / "data" / "tables.jsonl", "collection": None, "text_format": None, "embed_ckpt": None,
    "cond_filter": True, "pool": 50, "rerank_ckpt": None, "rerank_mode": "blend", "rerank_n": 50, "stage2": "none",
    "stage2_k": 20, "answer_k": 5, "device": None, "hq_build": None, "hq_weight": 1.0, "hq_bm25": False,
}
# where each old field lives now
NEW_PLACE = {"collection": lambda c: c.spec.chroma.collection, "tables": lambda c: c.spec.tables or files.TABLES}


@pytest.fixture
def env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _read(cfg: SearchConfig, name: str):
    if name in NEW_PLACE:
        return NEW_PLACE[name](cfg)
    return getattr(cfg.options if name in SearchOptions.model_fields else cfg.spec, name)


def test_empty_environment_gives_the_old_defaults(env):
    cfg = search_config_from_env()
    assert {name: _read(cfg, name) for name in OLD_DEFAULTS} == OLD_DEFAULTS
    assert (cfg.spec.chroma.host, cfg.spec.chroma.port) == ("localhost", 8001)  # env_chroma_config's defaults
    assert cfg.stage2_llm is None
    assert cfg.options == SearchOptions() and cfg.spec == IndexSpec()  # the objects' own defaults agree too
    assert cfg.options.label(cfg.spec) == "hybrid+cond"


def test_variables_are_converted_as_from_env_did(env):
    for key, value in {"CHROMA_HOST": "chroma", "CHROMA_PORT": "8000", "CHROMA_COLLECTION": "kosis_a_x",
                       "TEXT_FORMAT": "n2-L60", "EMBED_CKPT": "e.pt", "RERANK_CKPT": "r.pt", "RERANK_MODE": "score",
                       "STAGE2": " HCX ", "COND_FILTER": "off", "ANSWER_K": "7", "HQ_BUILD": " hq-x ",
                       "HQ_WEIGHT": "0.5", "HQ_BM25": "on", "HCX_API_KEY": "k-hcx", "HCX_THINKING": "LOW"}.items():
        env.setenv(key, value)
    cfg = search_config_from_env()
    assert (cfg.spec.chroma.host, cfg.spec.chroma.port, cfg.spec.chroma.collection) == ("chroma", 8000, "kosis_a_x")
    assert (cfg.spec.text_format, cfg.spec.embed_ckpt, cfg.spec.rerank_ckpt, cfg.spec.hq_build) == (
        "n2-L60", "e.pt", "r.pt", "hq-x")
    o = cfg.options
    assert (o.rerank_mode, o.stage2, o.cond_filter, o.answer_k, o.hq_weight, o.hq_bm25) == (
        "score", "hcx", False, 7, 0.5, True)
    assert (cfg.stage2_llm.provider, cfg.stage2_llm.model, cfg.stage2_llm.thinking) == ("hcx", "HCX-007", "low")
    assert require_key(cfg.stage2_llm) == "k-hcx"
    assert o.label(cfg.spec) == "hybrid+hq+hqbm25(w0.5) -> stage1-score -> hcx"


@pytest.mark.parametrize("stage2, provider, key_env", [("clova", "clova_rerank", "HCX_API_KEY"),
                                                       ("hcx", "hcx", "HCX_API_KEY"), ("gpt", "openai", "OPENAI_API_KEY")])
def test_stage2_provider_and_key(env, stage2, provider, key_env):
    env.setenv("STAGE2", stage2)
    assert search_config_from_env().stage2_llm.api_key is None  # no key: rerank2 raises LLMKeyMissing
    env.setenv(key_env, "secret")
    llm = search_config_from_env().stage2_llm
    assert llm.provider == provider and require_key(llm) == "secret"


def test_empty_variables_are_unset_and_overrides_win(env):
    for key in ("RERANK_CKPT", "HQ_BUILD", "STAGE2", "COND_FILTER", "ANSWER_K"):
        env.setenv(key, "")  # cmd `set X=` deletes, a shell `X=` empties: both mean unset
    cfg = search_config_from_env()
    assert (cfg.spec.rerank_ckpt, cfg.spec.hq_build, cfg.options.stage2, cfg.options.cond_filter,
            cfg.options.answer_k) == (None, None, "none", True, 5)
    env.setenv("RERANK_CKPT", "r.pt")
    cfg = search_config_from_env(cond_filter=False, rerank_ckpt=None, stage2="none", collection="c", pool=30)
    assert (cfg.options.cond_filter, cfg.spec.rerank_ckpt, cfg.spec.chroma.collection, cfg.options.pool) == (
        False, None, "c", 30)
    with pytest.raises(TypeError, match="unknown search setting"):
        search_config_from_env(nope=1)


def test_the_result_is_read_by_name():
    assert SearchConfig._fields == ("options", "spec", "stage2_llm", "guide_llm")


def test_guide_defaults_are_the_phase1_model_yaml_values(env):
    guide = search_config_from_env().guide_llm
    assert (guide.provider, guide.model, guide.temperature, guide.max_tokens, guide.timeout, guide.api_key) == (
        "clova_chat", "HCX-DASH-002", 0, 1024, 60, None)
    assert (config.GUIDE_MODEL, config.GUIDE_TEMPERATURE, config.GUIDE_MAX_TOKENS, config.GUIDE_TIMEOUT) == (
        "HCX-DASH-002", 0.0, 1024, 60.0)


def test_guide_key_falls_back_to_clovastudio_and_settings_follow_the_environment(env):
    env.setenv("CLOVASTUDIO_API_KEY", "k-studio")
    assert require_key(search_config_from_env().guide_llm) == "k-studio"
    env.setenv("HCX_API_KEY", "k-hcx")
    assert require_key(search_config_from_env().guide_llm) == "k-hcx"  # HCX_API_KEY first, as chain.clova_chat
    for key, value in {"GUIDE_MODEL": "HCX-005", "GUIDE_TEMPERATURE": "0.3", "GUIDE_MAX_TOKENS": "256",
                       "GUIDE_TIMEOUT": "10"}.items():
        env.setenv(key, value)
    guide = search_config_from_env().guide_llm
    assert (guide.model, guide.temperature, guide.max_tokens, guide.timeout) == ("HCX-005", 0.3, 256, 10.0)


def test_config_reads_no_dotenv():
    import ast

    tree = ast.parse(Path(config.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.startswith("dotenv") for m in imported)
