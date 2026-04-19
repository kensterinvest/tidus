#!/usr/bin/env python3
"""Phase 1 Recipe A — LoRA-on-DeBERTa-v3-xsmall multi-head classifier.

Ported from vLLM Semantic Router's `ft_linear_lora.py` with structured
deviations (documented below) to match Tidus plan.md requirements:

  vLLM SR (source)            ->  Tidus (port)
  --------------------------      --------------------------
  AutoModelForSequenceClassif  -> AutoModel + 3 nn.Linear heads
  single head, CE loss         -> 3 heads, summed CE with privacy 2x weight
  target_modules: query/value  -> query_proj/key_proj/value_proj/dense
                                  (DebertaV2 naming, verified via named_modules)

LoRA config (plan.md):
  rank 16-32, alpha = 2*rank, dropout 0.1, bias=none

Training (plan.md):
  AdamW, lr 2e-5..3e-5, weight_decay 0.1, cosine schedule, warmup 0.06,
  grad_accum 2, grad_clip 1.0, 3-5 epochs, batch 8-32,
  load_best_model_at_end=True

Usage (CPU-only, no CUDA available):
  uv run python scripts/train_encoder.py --smoke-test     # ~90s sanity check
  uv run python scripts/train_encoder.py                  # full train (~20-40 min)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train_encoder")

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "tests" / "classification" / "chunks"
POOL_DIR = REPO_ROOT / "tests" / "classification" / "pool_chunks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tidus" / "classification" / "weights"

BACKBONE = "microsoft/deberta-v3-xsmall"
MAX_LEN = 384  # prompts in pool are truncated at 2000 chars; 384 tokens fits most

DOMAINS = ["chat", "code", "reasoning", "extraction", "classification", "summarization", "creative"]
COMPLEXITIES = ["simple", "moderate", "complex", "critical"]
PRIVACIES = ["public", "internal", "confidential"]

DOM2IDX = {v: i for i, v in enumerate(DOMAINS)}
CMP2IDX = {v: i for i, v in enumerate(COMPLEXITIES)}
PRV2IDX = {v: i for i, v in enumerate(PRIVACIES)}

SEED = 42


@dataclass
class Row:
    text: str
    domain: int
    complexity: int
    privacy: int


def _load_overrides() -> dict[str, dict]:
    """Load label overrides from tests/classification/label_overrides.jsonl
    and (if present) tests/classification/label_overrides_irr.jsonl.

    Each line: {"id": ..., "privacy": ..., [optional other fields], "reason": ...}
    Overrides are applied in load_joined_rows() after the main label read.
    Introduced 2026-04-19 after the POC backtest eyeball revealed fictional-
    character medical narratives were mislabeled as confidential.

    IRR overrides added 2026-04-20 from cross-family adjudication
    (Claude + GPT + Gemini); later file wins on id collisions.
    """
    overrides: dict[str, dict] = {}
    for fname in ("label_overrides.jsonl", "label_overrides_irr.jsonl"):
        path = REPO_ROOT / "tests" / "classification" / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                overrides[r["id"]] = r
    return overrides


def load_joined_rows() -> list[Row]:
    """Join labels_*.jsonl to pool_*.jsonl on id, then apply label_overrides.jsonl.

    Skips unmatched ids (labels_001-009 predate the current pool).
    """
    pool_text: dict[str, str] = {}
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]

    overrides = _load_overrides()
    rows: list[Row] = []
    for lf in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] not in pool_text:
                    continue
                if r["id"] in overrides:
                    ov = overrides[r["id"]]
                    for k in ("domain", "complexity", "privacy"):
                        if k in ov:
                            r[k] = ov[k]
                rows.append(Row(
                    text=pool_text[r["id"]],
                    domain=DOM2IDX[r["domain"]],
                    complexity=CMP2IDX[r["complexity"]],
                    privacy=PRV2IDX[r["privacy"]],
                ))
    return rows


def balanced_class_weights(y: list[int], num_classes: int) -> torch.Tensor:
    """sklearn's 'balanced' formula: n_samples / (n_classes * bincount)."""
    counts = np.bincount(np.array(y), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0  # guard: zero-count classes get neutral weight
    weights = len(y) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class MultiHeadDeBERTa(nn.Module):
    """DeBERTa-v3 backbone with LoRA + three class-weighted classification heads.

    forward() returns a HuggingFace-Trainer-compatible dict with "loss" +
    per-head logits. Each head uses class-weighted CrossEntropy (sklearn's
    'balanced' formula) to counter the 85/11/4 privacy imbalance — without
    class weights the model degenerates to "always predict public" (observed
    in the first Recipe A run: confidential recall = 0 across all epochs).
    """

    def __init__(
        self,
        backbone_name: str,
        lora_cfg: LoraConfig,
        domain_weights: torch.Tensor,
        complexity_weights: torch.Tensor,
        privacy_weights: torch.Tensor,
    ):
        super().__init__()
        base = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float32)
        self.backbone = get_peft_model(base, lora_cfg)
        hidden = base.config.hidden_size
        self.domain_head = nn.Linear(hidden, len(DOMAINS))
        self.complexity_head = nn.Linear(hidden, len(COMPLEXITIES))
        self.privacy_head = nn.Linear(hidden, len(PRIVACIES))
        self.register_buffer("domain_weights", domain_weights)
        self.register_buffer("complexity_weights", complexity_weights)
        self.register_buffer("privacy_weights", privacy_weights)

    def forward(
        self,
        input_ids,
        attention_mask,
        domain_labels=None,
        complexity_labels=None,
        privacy_labels=None,
        **_ignore,
    ):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :].to(self.domain_head.weight.dtype)

        d_logits = self.domain_head(pooled)
        c_logits = self.complexity_head(pooled)
        p_logits = self.privacy_head(pooled)

        loss = None
        if domain_labels is not None:
            d_loss = nn.functional.cross_entropy(d_logits, domain_labels, weight=self.domain_weights)
            c_loss = nn.functional.cross_entropy(c_logits, complexity_labels, weight=self.complexity_weights)
            p_loss = nn.functional.cross_entropy(p_logits, privacy_labels, weight=self.privacy_weights)
            loss = d_loss + c_loss + p_loss

        return {
            "loss": loss,
            "domain_logits": d_logits,
            "complexity_logits": c_logits,
            "privacy_logits": p_logits,
        }


def build_dataset(rows: list[Row], tokenizer, max_len: int) -> Dataset:
    ds = Dataset.from_list([
        {
            "text": r.text,
            "domain_labels": r.domain,
            "complexity_labels": r.complexity,
            "privacy_labels": r.privacy,
        }
        for r in rows
    ])
    def tok(ex):
        enc = tokenizer(ex["text"], truncation=True, max_length=max_len, padding="max_length")
        ex["input_ids"] = enc["input_ids"]
        ex["attention_mask"] = enc["attention_mask"]
        return ex
    ds = ds.map(tok, remove_columns=["text"])
    ds.set_format(type="torch", columns=[
        "input_ids", "attention_mask",
        "domain_labels", "complexity_labels", "privacy_labels",
    ])
    return ds


def compute_metrics_factory():
    """Returns a compute_metrics fn reporting per-head accuracy + privacy confidential recall."""
    def compute(eval_pred):
        preds, labels = eval_pred
        # preds is a tuple: (domain_logits, complexity_logits, privacy_logits)
        # labels is a tuple with same shape per the Trainer's label_names wiring
        d_logits, c_logits, p_logits = preds
        d_true, c_true, p_true = labels

        d_pred = np.argmax(d_logits, axis=-1)
        c_pred = np.argmax(c_logits, axis=-1)
        p_pred = np.argmax(p_logits, axis=-1)

        m = {
            "domain_acc": float((d_pred == d_true).mean()),
            "complexity_acc": float((c_pred == c_true).mean()),
            "privacy_acc": float((p_pred == p_true).mean()),
            "domain_f1_macro": f1_score(d_true, d_pred, average="macro", zero_division=0),
            "privacy_f1_macro": f1_score(p_true, p_pred, average="macro", zero_division=0),
        }
        # Confidential recall — THE gate metric
        conf_idx = PRV2IDX["confidential"]
        conf_mask = p_true == conf_idx
        if conf_mask.sum() > 0:
            m["confidential_recall"] = float((p_pred[conf_mask] == conf_idx).mean())
        else:
            m["confidential_recall"] = 0.0
        return m
    return compute


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny 100-example / 1-epoch run to verify pipeline (~90s)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = load_joined_rows()
    log.info(f"Loaded {len(rows)} joined label rows")
    log.info(f"  Privacy dist: {Counter(PRIVACIES[r.privacy] for r in rows)}")
    log.info(f"  Domain dist:  {Counter(DOMAINS[r.domain] for r in rows)}")

    if args.smoke_test:
        rows = rows[:100]
        args.epochs = 1
        log.info(f"Smoke test: capped to {len(rows)} rows, 1 epoch")

    # Stratified split on privacy (rarest class — 4.3% confidential)
    privacy_labels = [r.privacy for r in rows]
    train_rows, val_rows = train_test_split(
        rows, test_size=0.15, random_state=SEED,
        stratify=privacy_labels if len(set(privacy_labels)) > 1 else None,
    )
    log.info(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    log.info(f"Loading tokenizer + backbone: {BACKBONE}")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    # Class-weighted CE per head (balanced formula from train labels only, no leakage)
    d_weights = balanced_class_weights([r.domain for r in train_rows], len(DOMAINS))
    c_weights = balanced_class_weights([r.complexity for r in train_rows], len(COMPLEXITIES))
    p_weights = balanced_class_weights([r.privacy for r in train_rows], len(PRIVACIES))
    log.info(f"Domain class weights:     {d_weights.tolist()}")
    log.info(f"Complexity class weights: {c_weights.tolist()}")
    log.info(f"Privacy class weights:    {p_weights.tolist()}")

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
    )
    model = MultiHeadDeBERTa(BACKBONE, lora_cfg, d_weights, c_weights, p_weights)
    model.backbone.print_trainable_parameters()

    log.info("Tokenizing datasets")
    train_ds = build_dataset(train_rows, tokenizer, args.max_len)
    val_ds = build_dataset(val_rows, tokenizer, args.max_len)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="epoch" if not args.smoke_test else "no",
        save_strategy="epoch" if not args.smoke_test else "no",
        load_best_model_at_end=not args.smoke_test,
        metric_for_best_model="privacy_f1_macro" if not args.smoke_test else None,
        greater_is_better=True,
        report_to="none",
        label_names=["domain_labels", "complexity_labels", "privacy_labels"],
        seed=SEED,
        dataloader_num_workers=0,
        fp16=False,  # CPU-only; FP32 for stable training per vLLM SR note
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_factory() if not args.smoke_test else None,
    )

    log.info("Starting training")
    trainer.train()

    # Final eval even in smoke test
    if args.smoke_test:
        log.info("Smoke-test final-eval (compute_metrics not wired in smoke mode; loss-only check)")
        eval_out = trainer.evaluate()
        log.info(f"Smoke-test eval_loss: {eval_out.get('eval_loss'):.4f}")
    else:
        log.info("Running final eval")
        final_eval = trainer.evaluate()
        for k, v in final_eval.items():
            log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save: LoRA adapter + heads + tokenizer + label mappings
    log.info(f"Saving artifacts to {output_dir}")
    model.backbone.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    torch.save({
        "domain_head": model.domain_head.state_dict(),
        "complexity_head": model.complexity_head.state_dict(),
        "privacy_head": model.privacy_head.state_dict(),
    }, output_dir / "heads.pt")
    (output_dir / "label_mappings.json").write_text(json.dumps({
        "domains": DOMAINS,
        "complexities": COMPLEXITIES,
        "privacies": PRIVACIES,
        "backbone": BACKBONE,
        "max_len": args.max_len,
        "class_weights": {
            "domain": d_weights.tolist(),
            "complexity": c_weights.tolist(),
            "privacy": p_weights.tolist(),
        },
    }, indent=2))
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
