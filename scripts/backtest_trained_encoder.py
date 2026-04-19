#!/usr/bin/env python3
"""Backtest the trained Recipe A encoder against the Phase 0 gate.

Reloads the LoRA adapter + classification heads saved by `train_encoder.py`,
runs inference on the same stratified val split the trainer measured on
(SEED=42 preserves the split), and applies the same pre-committed decision
rule as `backtest_poc_wildchat.py`:

    Wilson 95% CI lower bound >= gate  -> PASS
    Wilson 95% CI upper bound <  gate  -> FAIL
    CI straddles gate                   -> INCONCLUSIVE

Gates (plan.md / boot):
    Domain accuracy          >= 85%
    Confidential recall      >= 95%

Also reports metrics on the full 1569 labels (train + val) for diagnostic
comparison against the POC baseline — but *decision* is based on val only
to avoid leakage.

Usage:
    uv run python scripts/backtest_trained_encoder.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_encoder import (  # noqa: E402
    COMPLEXITIES,
    DOMAINS,
    PRIVACIES,
    PRV2IDX,
    SEED,
    Row,
    load_joined_rows,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "tidus" / "classification" / "weights"
CONF_IDX = PRV2IDX["confidential"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)


def gate_verdict(lo: float, hi: float, threshold: float) -> str:
    if lo >= threshold:
        return f"PASS (CI lower {lo*100:.1f}% >= {threshold*100:.0f}%)"
    if hi < threshold:
        return f"FAIL (CI upper {hi*100:.1f}% < {threshold*100:.0f}%)"
    return f"INCONCLUSIVE (CI [{lo*100:.1f}, {hi*100:.1f}] straddles {threshold*100:.0f}%)"


class TrainedEncoder:
    """Reload MultiHeadDeBERTa from the weights directory saved by train_encoder.py."""

    def __init__(self, weights_dir: Path):
        mappings = json.loads((weights_dir / "label_mappings.json").read_text())
        self.backbone_name = mappings["backbone"]
        self.max_len = mappings["max_len"]

        base = AutoModel.from_pretrained(self.backbone_name, torch_dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, weights_dir / "adapter")
        self.model.train(False)  # inference mode

        hidden = base.config.hidden_size
        self.domain_head = nn.Linear(hidden, len(DOMAINS))
        self.complexity_head = nn.Linear(hidden, len(COMPLEXITIES))
        self.privacy_head = nn.Linear(hidden, len(PRIVACIES))
        heads = torch.load(weights_dir / "heads.pt", map_location="cpu", weights_only=True)
        self.domain_head.load_state_dict(heads["domain_head"])
        self.complexity_head.load_state_dict(heads["complexity_head"])
        self.privacy_head.load_state_dict(heads["privacy_head"])
        for h in (self.domain_head, self.complexity_head, self.privacy_head):
            h.train(False)

        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir / "adapter")

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        pooled = out.last_hidden_state[:, 0, :].to(self.domain_head.weight.dtype)
        d = self.domain_head(pooled).argmax(dim=-1).cpu().numpy()
        c = self.complexity_head(pooled).argmax(dim=-1).cpu().numpy()
        p = self.privacy_head(pooled).argmax(dim=-1).cpu().numpy()
        return d, c, p


def run_inference(enc: TrainedEncoder, rows: list[Row], batch_size: int = 8
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_all, c_all, p_all = [], [], []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        d, c, p = enc.predict_batch([r.text for r in batch])
        d_all.append(d)
        c_all.append(c)
        p_all.append(p)
        if (i // batch_size) % 10 == 0:
            print(f"  ...{i + len(batch)}/{len(rows)}")
    return np.concatenate(d_all), np.concatenate(c_all), np.concatenate(p_all)


def report_slice(name: str, rows: list[Row],
                 d_pred: np.ndarray, c_pred: np.ndarray, p_pred: np.ndarray,
                 apply_gate: bool) -> tuple[bool, bool]:
    n = len(rows)
    d_true = np.array([r.domain for r in rows])
    c_true = np.array([r.complexity for r in rows])
    p_true = np.array([r.privacy for r in rows])

    dom_ok = int((d_pred == d_true).sum())
    cmp_ok = int((c_pred == c_true).sum())
    conf_mask = p_true == CONF_IDX
    conf_ground = int(conf_mask.sum())
    conf_flagged = int(((p_pred == CONF_IDX) & conf_mask).sum())
    conf_predicted = int((p_pred == CONF_IDX).sum())

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {name}  (n={n})")
    print(sep)

    dp, dlo, dhi = wilson_ci(dom_ok, n)
    print(f"\n  Domain accuracy: {dom_ok}/{n} = {dp*100:.2f}%   CI [{dlo*100:.2f}, {dhi*100:.2f}]")
    if apply_gate:
        print(f"    Gate >= 85%  -> {gate_verdict(dlo, dhi, 0.85)}")

    cp, clo, chi = wilson_ci(cmp_ok, n)
    print(f"\n  Complexity accuracy: {cmp_ok}/{n} = {cp*100:.2f}%   CI [{clo*100:.2f}, {chi*100:.2f}]")

    dom_pass = dlo >= 0.85
    dom_fail = dhi < 0.85

    if conf_ground:
        pp, plo, phi = wilson_ci(conf_flagged, conf_ground)
        print(f"\n  Confidential recall: {conf_flagged}/{conf_ground} = {pp*100:.2f}%   CI [{plo*100:.2f}, {phi*100:.2f}]")
        if apply_gate:
            print(f"    Gate >= 95%  -> {gate_verdict(plo, phi, 0.95)}")
        conf_pass = plo >= 0.95
        conf_fail = phi < 0.95
    else:
        print("\n  Confidential recall: n/a (0 confidentials in slice)")
        conf_pass = False
        conf_fail = False

    if conf_predicted:
        prec = conf_flagged / conf_predicted
        print(f"  Confidential precision: {conf_flagged}/{conf_predicted} = {prec*100:.2f}%")

    print("\n  Privacy confusion (rows=actual, cols=predicted)")
    print("    " + " ".join(f"{v[:7]:>7}" for v in PRIVACIES) + "   | total")
    for ai, aname in enumerate(PRIVACIES):
        mask = p_true == ai
        tot = int(mask.sum())
        row = f"    {aname[:13]:>13}"
        for pi in range(len(PRIVACIES)):
            c = int(((p_pred == pi) & mask).sum())
            row += f" {c:>6d} " if c else "      . "
        print(row + f"  | {tot}")

    if apply_gate:
        print(f"\n  Gate decision: Domain {'PASS' if dom_pass else ('FAIL' if dom_fail else 'INCONCLUSIVE')}  |  "
              f"Privacy {'PASS' if conf_pass else ('FAIL' if conf_fail else 'INCONCLUSIVE')}")
    return (dom_pass and conf_pass, dom_fail or conf_fail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    if not (weights_dir / "label_mappings.json").exists():
        print(f"ERROR: no trained weights at {weights_dir}.")
        print("Run `uv run python scripts/train_encoder.py` first.")
        return 1

    print(f"Loading trained encoder from {weights_dir}")
    enc = TrainedEncoder(weights_dir)

    rows = load_joined_rows()
    print(f"Joined {len(rows)} label rows")
    print(f"  Privacy dist: {Counter(PRIVACIES[r.privacy] for r in rows)}")

    privacy_labels = [r.privacy for r in rows]
    train_rows, val_rows = train_test_split(
        rows, test_size=0.15, random_state=SEED, stratify=privacy_labels,
    )
    print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}")

    print("\nRunning inference on val set (the gate-decision slice)...")
    d_val, c_val, p_val = run_inference(enc, val_rows, args.batch_size)

    print("\nRunning inference on full set (diagnostic, overfits train)...")
    d_all, c_all, p_all = run_inference(enc, rows, args.batch_size)

    val_pass, val_fail = report_slice("VAL SLICE (gate decision)", val_rows,
                                       d_val, c_val, p_val, apply_gate=True)
    report_slice("FULL SET (diagnostic — includes train, not for gate)", rows,
                 d_all, c_all, p_all, apply_gate=False)

    print("\n" + "=" * 70)
    print("  PHASE 0 GATE (val-slice decision):")
    if val_pass:
        print("  -> PASS: encoder may ship to Tier 2.")
        return 0
    if val_fail:
        print("  -> FAIL: iterate on labels (quality audit) or escalate to Recipe B.")
        return 1
    print("  -> INCONCLUSIVE: val CI straddles gate — need more labels or k-fold.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
