"""Stage 2 selector factories (T2-41 S3: from rerank.py): name -> (text_of, config: LLMConfig) -> Select.

    clova  CLOVA Studio reranker API (llm.clova_rerank.ClovaReranker)             provider clova_rerank
    hcx    HCX-007 picks numbered tables, may pick none (rerank.hcx.HcxSelector)  provider hcx
    gpt    the same prompt and parser over OpenAI (HcxSelector + llm.openai)      provider openai

The key is in the config (LLMConfig.api_key); a selector reads it at its first call, and the rerank2 node refuses a
config without one before that (LLMKeyMissing). Another LLM re-ranker is one more entry with the same signature.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import SecretStr

from kosis_rag.llm.clova_rerank import ClovaReranker
from kosis_rag.llm.config import LLMConfig
from kosis_rag.llm.openai import OpenAIClient
from kosis_rag.rerank.constants import CLOVA_RERANK_MODEL, GPT_MODEL, MODEL
from kosis_rag.rerank.hcx import HcxSelector
from kosis_rag.rerank.stage2 import Select

# stage 2 name -> (LLMConfig.provider, default model)
STAGE2_LLM: dict[str, tuple[str, str]] = {
    "clova": ("clova_rerank", CLOVA_RERANK_MODEL),
    "hcx": ("hcx", MODEL),
    "gpt": ("openai", GPT_MODEL),
}


def default_config(name: str, *, api_key: str | SecretStr | None = None, thinking: str = "none",
                   model: str | None = None, base_url: str | None = None) -> LLMConfig:
    """The LLMConfig a stage 2 name runs with by default (provider and model of STAGE2_LLM).

    In: name = a LLM_RERANKERS key; api_key = the provider's key (None: LLMKeyMissing at rerank2);
        thinking = HCX-007 thinking.effort; model = another model name (None: STAGE2_LLM's); base_url = an
        OpenAI-compatible server for "gpt" (None: OpenAI; T3-23). Out: LLMConfig. Side effects: KeyError for an
        unknown name. Node: none.
    """
    provider, default_model = STAGE2_LLM[name]
    return LLMConfig(provider=provider, api_key=api_key or None, model=model or default_model, thinking=thinking,
                     base_url=base_url or None)


def clova(text_of: dict[str, str], config: LLMConfig) -> Select:
    return ClovaReranker(text_of=text_of, config=config)


def hcx(text_of: dict[str, str], config: LLMConfig) -> Select:
    return HcxSelector(text_of=text_of, config=config)


def gpt(text_of: dict[str, str], config: LLMConfig) -> Select:
    return HcxSelector(text_of=text_of, config=config, client=OpenAIClient(config))


LLM_RERANKERS: dict[str, Callable[[dict[str, str], LLMConfig], Select]] = {
    "clova": clova,
    "hcx": hcx,
    "gpt": gpt,
}
