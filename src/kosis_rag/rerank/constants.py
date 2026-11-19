"""Constants of the rerank stage (T2-41 S3; from rerank.py and hcx.py). RRF_C is kosis_rag.shared.constants.RRF_C."""

from __future__ import annotations

MODEL = "HCX-007"  # the stage 2 HCX selector's model
MAX_PICK = 5  # tables the HCX selector may pick
# thinking tokens count toward maxCompletionTokens (Phase 1 configs/model.yaml): leave room when thinking is on
MAX_TOKENS = {"none": 512, "low": 5120, "medium": 5120, "high": 8192}
# v2 (T2-09 dev): v1 asked for tables "that directly hold the number" and refused even rank-1 gold tables
# (lab/T2-09-hcx-no-answer/prompt_v1.txt); v2 asks for the measure only and leaves years and region detail to code.
# v3 clarifies the selector contract and examples while preserving v2's measure-only decision boundary.
PROMPT_VERSION = "v4"
PROMPT_FILE = f"stage2-hcx-{PROMPT_VERSION}.md"  # rerank/prompts/<file>: the HCX selector's system prompt
# KOS-192 (opt-in, the rewrite path only): v4 plus how to read the user message's HINT_LABEL line; the graph and the
# default selector keep PROMPT_FILE
PROMPT_FILE_RW = "stage2-hcx-v5-rw.md"
HINT_LABEL = "검색용 표현"  # the user message line `검색용 표현: <rewrite>` under the question (hcx.user_message hint)
GPT_MODEL = "gpt-4o-mini"  # stage 2 "gpt": the same prompt and parser over OpenAI (T2-09 lab)
JEV_MODEL = "Cloudflare/clef-flash"  # stage 2 "jev": a Jev-style decision model over /v1/systemone (T3-23)
JEV_MODES = ("noul", "choice")
JEV_THRESHOLD = {"noul": 0.5, "choice": 0.05}  # default pick threshold per mode (choice: probabilities sum to 1)
# the jev prompt: rerank/prompts/stage2-jev-<mode>-<version>.md (rerank.systemone.jev_prompt); JEV_PROMPT picks another
JEV_PROMPT_VERSION = "v2"
CLOVA_RERANK_MODEL = "reranker"  # stage 2 "clova": the CLOVA Studio reranker API has no model name; a label only
