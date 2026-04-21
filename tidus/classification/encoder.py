"""Tier 2 encoder runtime — Recipe B (frozen MiniLM + sklearn LR heads).

Recipe B won over Recipe A (LoRA-on-DeBERTa) at the 2026-04 benchmark gate:
comparable macro-F1 at ~1/30th the training cost and no GPU required for
inference. This module wraps the trained artefacts for runtime classification.

Artefact layout in `weights_dir`:
    domain_head.joblib        sklearn LogisticRegression (class_weight='balanced')
    complexity_head.joblib    ditto
    privacy_head.joblib       ditto
    label_mappings.json       {"domains": [...], "complexities": [...],
                               "privacies": [...], "embed_model": str,
                               "max_chars": int, "recipe": "B"}

Inference is sync (SentenceTransformer uses torch). FastAPI handlers should
wrap `classify()` with `asyncio.to_thread` to keep the event loop free.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from tidus.classification.models import (
    EncoderLoadError,
    EncoderResult,
)

log = logging.getLogger(__name__)

# Project root (D:\dev\tidus) — lets us resolve relative config paths even
# when Tidus runs from a different CWD (systemd, docker, uvicorn --reload).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class _LRHead(Protocol):
    """Duck type for sklearn LogisticRegression we need."""

    def predict(self, X) -> np.ndarray: ...  # noqa: N803
    def predict_proba(self, X) -> np.ndarray: ...  # noqa: N803


class EncoderProtocol(Protocol):
    """Structural typing interface for what TaskClassifier needs from an encoder.

    The concrete `Encoder` below satisfies this automatically. Tests and future
    replacements (e.g., ONNX runtime, remote encoder) only need to match this
    shape — no subclassing required (backlog task #46).
    """

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def classify(self, text: str) -> EncoderResult: ...


class Encoder:
    """Frozen MiniLM encoder + three sklearn LR classification heads.

    Lifecycle:
        1. `__init__` — cheap, stores config
        2. `load()` — expensive (~2-5s), loads model + heads from disk
        3. `classify(text)` — fast (~5-15ms for short text), returns EncoderResult

    `load()` is idempotent and safe to call multiple times. Callers should
    invoke it once at service startup.
    """

    def __init__(
        self,
        weights_dir: Path | str,
        embed_model: str | None = None,
        max_chars: int | None = None,
    ) -> None:
        wd = Path(weights_dir)
        if not wd.is_absolute():
            wd = (_REPO_ROOT / wd).resolve()
        self._weights_dir = wd
        self._embed_model_override = embed_model
        self._max_chars_override = max_chars

        # Populated by load()
        self._embed = None
        self._domain_head: _LRHead | None = None
        self._complexity_head: _LRHead | None = None
        self._privacy_head: _LRHead | None = None
        self._domains: list[str] = []
        self._complexities: list[str] = []
        self._privacies: list[str] = []
        self._embed_model_name: str = ""
        self._max_chars: int = 1200
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def weights_dir(self) -> Path:
        return self._weights_dir

    def load(self) -> None:
        """Load SentenceTransformer + 3 joblib heads + label mappings.

        Raises EncoderLoadError on missing files or schema mismatch. The
        classifier catches this at startup and logs a clear diagnostic —
        service falls back to Tier-1-only safe defaults until weights exist.
        """
        if self._loaded:
            return

        mapping_path = self._weights_dir / "label_mappings.json"
        if not mapping_path.is_file():
            raise EncoderLoadError(
                f"label_mappings.json missing at {mapping_path}. "
                "Train via scripts/train_encoder_recipe_b.py.",
            )
        mappings = json.loads(mapping_path.read_text(encoding="utf-8"))

        # Validate the label spaces match our Literal types in models.py so
        # predictions never produce out-of-taxonomy values.
        self._domains = list(mappings["domains"])
        self._complexities = list(mappings["complexities"])
        self._privacies = list(mappings["privacies"])
        self._embed_model_name = self._embed_model_override or mappings["embed_model"]
        self._max_chars = self._max_chars_override or int(mappings.get("max_chars", 1200))

        for name, path in [
            ("domain_head",     self._weights_dir / "domain_head.joblib"),
            ("complexity_head", self._weights_dir / "complexity_head.joblib"),
            ("privacy_head",    self._weights_dir / "privacy_head.joblib"),
        ]:
            if not path.is_file():
                raise EncoderLoadError(f"{name} missing at {path}")

        self._domain_head     = joblib.load(self._weights_dir / "domain_head.joblib")
        self._complexity_head = joblib.load(self._weights_dir / "complexity_head.joblib")
        self._privacy_head    = joblib.load(self._weights_dir / "privacy_head.joblib")

        # Lazy-import SentenceTransformer — it pulls in torch and adds
        # ~300MB resident memory. Deferred until we actually need it so
        # that `from tidus.classification.encoder import Encoder` stays cheap.
        from sentence_transformers import SentenceTransformer

        log.info(
            "Loading encoder: model=%s weights=%s",
            self._embed_model_name, self._weights_dir,
        )
        self._embed = SentenceTransformer(self._embed_model_name)
        self._loaded = True

    def classify(self, text: str) -> EncoderResult:
        """Classify `text` across all three heads.

        Must be called after `load()`. Raises RuntimeError if not loaded —
        the classifier's own code path enforces load-ordering, so reaching
        this error means a programmer mistake, not a runtime condition.
        """
        if not self._loaded or self._embed is None:
            raise RuntimeError("Encoder.classify() called before load()")

        snippet = text[: self._max_chars]
        # normalize_embeddings=True matches how heads were trained.
        embedding = self._embed.encode(
            [snippet], normalize_embeddings=True, show_progress_bar=False,
        )

        domain, dom_conf = self._predict_with_confidence(
            self._domain_head, embedding, self._domains,
        )
        complexity, cmp_conf = self._predict_with_confidence(
            self._complexity_head, embedding, self._complexities,
        )
        privacy, prv_conf = self._predict_with_confidence(
            self._privacy_head, embedding, self._privacies,
        )

        # Surface the 384-dim embedding for Stage B telemetry to dim-reduce.
        # Converting to list[float] now keeps EncoderResult JSON-serialisable
        # without Pydantic's arbitrary_types_allowed. Cost: ~3 KB per call;
        # negligible vs the encode() compute itself.
        return EncoderResult(
            domain=domain,  # type: ignore[arg-type]
            complexity=complexity,  # type: ignore[arg-type]
            privacy=privacy,  # type: ignore[arg-type]
            confidence={
                "domain": dom_conf,
                "complexity": cmp_conf,
                "privacy": prv_conf,
            },
            embedding=embedding[0].astype(float).tolist(),
        )

    @staticmethod
    def _predict_with_confidence(
        head: _LRHead, embedding: np.ndarray, labels: list[str],
    ) -> tuple[str, float]:
        probs = head.predict_proba(embedding)[0]  # shape (n_classes,)
        idx = int(np.argmax(probs))
        return labels[idx], float(probs[idx])


def resolve_weights_dir(settings_dir: str) -> Path:
    """Resolve `settings.classify_encoder_dir` against project root."""
    p = Path(settings_dir)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p
