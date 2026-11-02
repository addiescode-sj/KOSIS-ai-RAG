"""Model loading for the fine-tuned search models (T2-06, T2-07). Inference only; training stays in min `lab/`.

Both fine-tunings trained only the top layers, so a checkpoint (`.pt`, ~400MB) holds just those layers. Loading
= the public base model + the checkpoint's tensors put back by name (`strict=False`, but a tensor the base model
does not have is an error, so a checkpoint for another model never loads silently). On a GPU the model runs in fp16.
Checkpoints are not in git: they live under data/models/ (gitignored) and `.env` points at them
(EMBED_CKPT, RERANK_CKPT). Without a checkpoint the base model loads as is (useful as a baseline).

A model is only right for the table text it was trained on (kosis_rag.text formats). The trainers write a `.json`
next to each `.pt`; its "doc_format" names that text, and loading with another text format is refused. Checkpoints
without the key were trained on "n2-L60" (plans 29 and 43).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMBED_BASE = "BAAI/bge-m3"
RERANK_BASE = "BAAI/bge-reranker-v2-m3"
DOC_TOKENS = 1024  # passage length the fine-tuned embedding was trained and measured with
QUERY_TOKENS = 512
PAIR_TOKENS = 512  # (question, table) pair length the re-ranker was trained with


def resolve_ckpt(path: str | Path | None) -> Path | None:
    """A `.pt` file, or a run folder holding `<name>-e<N>.pt` files of one name (its last epoch). None / "" -> None."""
    if not path:
        return None
    path = Path(path)
    if path.is_dir():
        pts = [p for p in path.glob("*-e*.pt") if p.stem.rsplit("-e", 1)[-1].isdigit()]
        if not pts:
            raise FileNotFoundError(f"no *-e<N>.pt checkpoint in {path}")
        names = {p.stem.rsplit("-e", 1)[0] for p in pts}
        if len(names) > 1:  # e.g. seen-e2.pt and unseen-e2.pt: the folder does not say which
            raise ValueError(f"{path} holds checkpoints of {sorted(names)}; point at the .pt file")
        path = max(pts, key=lambda p: int(p.stem.rsplit("-e", 1)[-1]))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def ckpt_id(path: Path | None) -> str:
    """'<file name>@<sha256[:12]>', recorded with a collection so queries are embedded with the same weights."""
    if path is None:
        return "base"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return f"{path.name}@{h.hexdigest()[:12]}"


LEGACY_FORMAT = "n2-L60"  # checkpoints written before the trainers recorded their text format


def trained_format(path: Path | None) -> str | None:
    """Text format a checkpoint was trained on (from its .json); None for the base model (no training)."""
    if path is None:
        return None
    meta = path.with_suffix(".json")
    if meta.is_file():
        return json.loads(meta.read_text(encoding="utf-8")).get("doc_format", LEGACY_FORMAT)
    return LEGACY_FORMAT


def check_format(path: Path | None, doc_format: str | None) -> None:
    trained = trained_format(path)
    if doc_format and trained and trained != doc_format:
        raise ValueError(f"{path.name} was trained on {trained!r} text, the search uses {doc_format!r}; "
                         "use weights trained on that text or set TEXT_FORMAT to match")


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def put_back(module: Any, ckpt: Path) -> int:
    """Copy the checkpoint's tensors into `module` (same dtype); returns how many were loaded."""
    import torch

    state = torch.load(ckpt, map_location="cpu")
    dtype = next(module.parameters()).dtype
    result = module.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"{ckpt.name} does not fit the base model: {result.unexpected_keys[:3]}")
    return len(state)


@dataclass
class Embedder:
    """Unit-length vectors (cosine = dot product) for passages and questions."""

    model: Any
    name: str  # base model name
    ckpt: str  # ckpt_id

    def _encode(self, texts: list[str], max_tokens: int, batch: int) -> list[list[float]]:
        self.model.max_seq_length = max_tokens
        return self.model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 100).tolist()

    def embed_documents(self, texts: list[str], batch: int = 16) -> list[list[float]]:
        return self._encode(texts, DOC_TOKENS, batch)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], QUERY_TOKENS, 1)[0]

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text)["input_ids"])


def load_embedder(ckpt: str | Path | None = None, base: str = EMBED_BASE, device: str | None = None,
                  doc_format: str | None = None) -> Embedder:
    from sentence_transformers import SentenceTransformer

    path = resolve_ckpt(ckpt)
    check_format(path, doc_format)
    model = SentenceTransformer(base, device=_device(device))
    if path is not None:
        put_back(model[0].auto_model, path)
    if model.device.type == "cuda":
        model.half()
    return Embedder(model=model, name=base, ckpt=ckpt_id(path))


@dataclass
class CrossEncoderScorer:
    """score(question, passages) -> one relevance per passage (sigmoid of the logit; only the order is used)."""

    model: Any
    name: str
    ckpt: str
    batch: int = 16

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        out = self.model.predict([(question, p) for p in passages], batch_size=self.batch,
                                 show_progress_bar=False, convert_to_numpy=True)
        return [float(v) for v in out.reshape(-1)]


def load_cross_encoder(ckpt: str | Path | None = None, base: str = RERANK_BASE, device: str | None = None,
                       max_length: int = PAIR_TOKENS, batch: int = 16, doc_format: str | None = None) -> CrossEncoderScorer:
    from sentence_transformers import CrossEncoder

    path = resolve_ckpt(ckpt)
    check_format(path, doc_format)
    device = _device(device)
    ce = CrossEncoder(base, max_length=max_length, device=device)
    hf = getattr(ce, "model", None) or ce
    if path is not None:
        put_back(hf, path)
    if device == "cuda":  # after loading, so the fp32 checkpoint is cast once
        hf.half()
    return CrossEncoderScorer(model=ce, name=base, ckpt=ckpt_id(path), batch=batch)
