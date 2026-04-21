"""Integration test — real Recipe B encoder on real prompts.

Loads the actual weights_b/ artefacts + SentenceTransformer (~5s). Pulls in
torch, so this sits in tests/integration/ alongside other I/O-heavy tests.

Skipped automatically when `tidus/classification/weights_b/` hasn't been
trained — CI environments without the weights still pass the rest of the
suite. Run locally after `uv run python scripts/train_encoder_recipe_b.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tidus.classification import TaskClassifier
from tidus.classification.encoder import Encoder, resolve_weights_dir
from tidus.classification.models import EncoderLoadError

_WEIGHTS_DIR = resolve_weights_dir("tidus/classification/weights_b")
_WEIGHTS_READY = (_WEIGHTS_DIR / "label_mappings.json").is_file() and all(
    (_WEIGHTS_DIR / n).is_file()
    for n in ("domain_head.joblib", "complexity_head.joblib", "privacy_head.joblib")
)

needs_weights = pytest.mark.skipif(
    not _WEIGHTS_READY,
    reason=f"Recipe B weights missing at {_WEIGHTS_DIR}; run scripts/train_encoder_recipe_b.py",
)


@pytest.fixture(scope="session")
def loaded_encoder() -> Encoder:
    enc = Encoder(weights_dir=str(_WEIGHTS_DIR))
    enc.load()
    return enc


@needs_weights
class TestEncoderLoadsAndPredicts:
    def test_load_is_idempotent(self, loaded_encoder: Encoder):
        loaded_encoder.load()  # second call
        assert loaded_encoder.loaded

    def test_code_prompt_classified(self, loaded_encoder: Encoder):
        r = loaded_encoder.classify(
            "Write a Python function that sorts a list using quicksort.",
        )
        assert r.domain in {
            "chat", "code", "reasoning", "extraction",
            "classification", "summarization", "creative",
        }
        assert r.complexity in {"simple", "moderate", "complex", "critical"}
        assert r.privacy in {"public", "internal", "confidential"}
        # Confidence is softmax max — must be in [0, 1].
        for k, v in r.confidence.items():
            assert 0.0 <= v <= 1.0, f"{k} conf out of range: {v}"

    def test_resume_prompt_recognized_confidential(self, loaded_encoder: Encoder):
        """A resume prompt (real name + employer + role) sits inside the
        encoder's training distribution for confidential — measured at
        ~0.69 confidence. A regression here (drift to internal/public)
        signals the privacy head's training-distribution recall has slipped.

        This test is NOT a privacy-recall metric — it asserts a single
        representative point. Full privacy recall is measured via
        `scripts/backtest_*.py` against the cross-family IRR corpus.
        """
        r = loaded_encoder.classify(
            "Review my resume. Sarah Chen, Senior Engineer at Stripe "
            "for 5 years, specializing in payments infrastructure and "
            "fraud detection. Looking for staff-level roles.",
        )
        assert r.privacy == "confidential", (
            f"Encoder regressed on training-distribution confidential prompt: "
            f"got {r.privacy} at confidence {r.confidence['privacy']:.3f}"
        )

    def test_benign_weather_prompt_recognized_public(self, loaded_encoder: Encoder):
        """Symmetric to the confidential test — an obviously-public prompt
        must not flag confidential. This catches the opposite regression.
        """
        r = loaded_encoder.classify("What's the weather like in Toronto today?")
        assert r.privacy != "confidential"


class TestEncoderErrorPaths:
    def test_missing_mapping_raises(self, tmp_path: Path):
        enc = Encoder(weights_dir=str(tmp_path))
        with pytest.raises(EncoderLoadError, match="label_mappings.json"):
            enc.load()

    def test_missing_head_raises(self, tmp_path: Path):
        (tmp_path / "label_mappings.json").write_text(
            '{"domains": ["chat"], "complexities": ["simple"],'
            ' "privacies": ["public"], "embed_model": "all-MiniLM-L6-v2",'
            ' "max_chars": 1200}',
        )
        enc = Encoder(weights_dir=str(tmp_path))
        with pytest.raises(EncoderLoadError, match="head missing"):
            enc.load()


@needs_weights
class TestClassifierWithRealEncoder:
    def test_classifier_startup_loads_encoder(self, loaded_encoder: Encoder):
        # Directly inject loaded encoder to skip startup()'s disk I/O.
        clf = TaskClassifier(encoder=loaded_encoder)
        r = clf.classify("What's 2 + 2?")
        assert r.classification_tier == "encoder"
        assert r.confidence["domain"] > 0
