"""The search settings of the app and the scripts, from the shell environment (T2-41 S4, §0.6).

The module (src/kosis_rag) reads no .env and no environment: the caller declares IndexSpec, SearchOptions and the
LLM configs. This is where the backend demo (and scripts/parity_l1.py) declare them: from os.environ only — no .env
is read (set the variables in the shell, docker-compose `environment:`, or a script you `call` / `source`).

    CHROMA_HOST CHROMA_PORT CHROMA_COLLECTION      Chroma server and the dense collection (default localhost:8001)
    TEXT_FORMAT EMBED_CKPT RERANK_CKPT RERANK_MODE  table text check, query embedder, stage 1 cross-encoder
    STAGE2 COND_FILTER ANSWER_K                     stage 2 LLM (clova | hcx | gpt | none), conditions, k default
    HQ_BUILD HQ_WEIGHT HQ_BM25                      T2-39 question index (empty HQ_BUILD = off)
    HCX_API_KEY HCX_THINKING OPENAI_API_KEY         stage 2 keys (clova / hcx: HCX_API_KEY, gpt: OPENAI_API_KEY)
    GUIDE_MODEL GUIDE_TEMPERATURE GUIDE_MAX_TOKENS GUIDE_TIMEOUT   the guide LLM (ChatClovaX)
    HCX_API_KEY, else CLOVASTUDIO_API_KEY           the guide LLM's key (None: the guide node answers 503)

The keys and defaults are the ones SearchSettings.from_env read (tests/test_config_from_env.py compares them field by
field). The guide defaults are Phase 1 configs/model.yaml @e248a45 written as constants (that file is no longer read).
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

from kosis_rag.llm.config import LLMConfig
from kosis_rag.rerank import selectors
from kosis_rag.schemas.resources import OFF, ChromaConfig, IndexSpec, SearchOptions

# The guide model: Phase 1 configs/model.yaml @e248a45 (model.name, temperature, max_tokens, timeout_seconds)
GUIDE_MODEL = "HCX-DASH-002"
GUIDE_TEMPERATURE = 0.0
GUIDE_MAX_TOKENS = 1024
GUIDE_TIMEOUT = 60.0
GUIDE_KEY_ENV = ("HCX_API_KEY", "CLOVASTUDIO_API_KEY")  # first set one wins (ChatClovaX's own fallback, now explicit)
# The environment variable holding each stage 2 LLM's key
STAGE2_KEY_ENV = {"clova": "HCX_API_KEY", "hcx": "HCX_API_KEY", "gpt": "OPENAI_API_KEY"}
CHROMA_FIELDS = {"collection": "collection"}  # override name -> ChromaConfig field (SearchSettings had `collection`)


class SearchConfig(NamedTuple):
    """What search_config_from_env gives. Read it by name (cfg.spec, cfg.options): the order is options first."""

    options: SearchOptions
    spec: IndexSpec
    stage2_llm: LLMConfig | None  # None when stage 2 is off or names no known re-ranker
    guide_llm: LLMConfig


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


def stage2_llm_from_env(name: str) -> LLMConfig | None:
    """The stage 2 LLM's config: selectors.default_config(name) with its key (STAGE2_KEY_ENV) and HCX_THINKING.

    In: name = options.stage2. Out: the config, or None for none / an unknown name (load_resources then raises).
    Side effects: reads os.environ. Node: none.
    """
    if name in OFF or name not in selectors.LLM_RERANKERS:
        return None
    return selectors.default_config(name, api_key=_env(STAGE2_KEY_ENV[name]),
                                    thinking=(_env("HCX_THINKING") or "none").strip().lower())


def guide_llm_from_env() -> LLMConfig:
    """The guide LLM (ChatClovaX) config: GUIDE_* or the model.yaml defaults; key HCX_API_KEY, else
    CLOVASTUDIO_API_KEY, else None (Runtime.guide_client raises LLMKeyMissing). Side effects: reads os.environ."""
    key = next((value for value in map(_env, GUIDE_KEY_ENV) if value), None)
    return LLMConfig(provider="clova_chat", api_key=key, model=_env("GUIDE_MODEL") or GUIDE_MODEL,
                     temperature=float(_env("GUIDE_TEMPERATURE") or GUIDE_TEMPERATURE),
                     max_tokens=int(_env("GUIDE_MAX_TOKENS") or GUIDE_MAX_TOKENS),
                     timeout=float(_env("GUIDE_TIMEOUT") or GUIDE_TIMEOUT))


def search_config_from_env(**overrides: Any) -> SearchConfig:
    """The search settings from the shell environment, with overrides (the old SearchSettings.from_env(**overrides)).

    In: overrides = any SearchOptions or IndexSpec field (a whole `chroma` replaces CHROMA_HOST / CHROMA_PORT /
        CHROMA_COLLECTION), or `collection` (the Chroma collection), given by name;
        they win over the environment (e.g. the L1 presets: cond_filter=True, rerank_ckpt=None, stage2="none").
    Out: SearchConfig(options, spec, stage2_llm, guide_llm) — pass them by name:
        load_resources(spec=cfg.spec, options=cfg.options, llm=cfg.stage2_llm).
    Side effects: reads os.environ (never .env); TypeError for an unknown override. Node: none.
    """
    env = _env
    values: dict[str, Any] = {
        "collection": env("CHROMA_COLLECTION"),
        "text_format": env("TEXT_FORMAT"),
        "embed_ckpt": env("EMBED_CKPT"),
        "cond_filter": (env("COND_FILTER") or "on").strip().lower() not in OFF,
        "rerank_ckpt": env("RERANK_CKPT"),
        "rerank_mode": env("RERANK_MODE") or "blend",
        "stage2": (env("STAGE2") or "none").strip().lower(),
        "answer_k": int(env("ANSWER_K") or 5),
        "hq_build": (env("HQ_BUILD") or "").strip() or None,
        "hq_weight": float(env("HQ_WEIGHT") or 1.0),
        "hq_bm25": (env("HQ_BM25") or "off").strip().lower() not in OFF,
    }
    values.update(overrides)
    unknown = set(values) - set(SearchOptions.model_fields) - set(IndexSpec.model_fields) - set(CHROMA_FIELDS)
    if unknown:
        raise TypeError(f"unknown search setting(s): {sorted(unknown)}")
    chroma = values.pop("chroma", None) or ChromaConfig(
        host=env("CHROMA_HOST") or "localhost", port=int(env("CHROMA_PORT") or 8001),
        **{CHROMA_FIELDS[k]: v for k, v in values.items() if k in CHROMA_FIELDS})
    options = SearchOptions(**{k: v for k, v in values.items() if k in SearchOptions.model_fields})
    spec = IndexSpec(chroma=chroma, **{k: v for k, v in values.items() if k in IndexSpec.model_fields})
    return SearchConfig(options=options, spec=spec, stage2_llm=stage2_llm_from_env(options.stage2),
                        guide_llm=guide_llm_from_env())
