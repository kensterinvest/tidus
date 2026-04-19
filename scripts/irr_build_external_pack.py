#!/usr/bin/env python3
"""Build IRR labeling packs for cross-family rater agreement study.

Samples 150 stratified rows (70 confidential + 40 internal + 40 public) with
the 3 structural-miss audit cases force-included, and writes blind labeling
packs that external labelers (GPT, Gemini, etc.) can classify via copy-paste.

Outputs all to tests/classification/irr/:
  - rubric.md               (self-contained rubric for external labelers)
  - pack_01.md..pack_10.md  (10 blind packs, 15 prompts each)
  - truth_keys.jsonl        (our labels, keyed by id — NEVER shown to labelers)
  - README.md               (step-by-step user workflow)
  - sample_manifest.jsonl   (metadata: which ids appear in which pack)

Run:  uv run python scripts/irr_build_external_pack.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "tests" / "classification" / "chunks"
POOL_FILE = ROOT / "tests" / "classification" / "prompts_pool.jsonl"
OVERRIDES_FILE = ROOT / "tests" / "classification" / "label_overrides.jsonl"
IRR_DIR = ROOT / "tests" / "classification" / "irr"

SEED = 20260419
# Note: take ALL unique post-override confidentials (natural n, ~69-70 from
# master pool with 1-2 orphans). Add 40 internal + 40 public for sanity classes.
# Ensemble-reported "71" was inflated by 7 duplicate label lines; true unique
# confidential count is lower.
N_INTERNAL = 40
N_PUBLIC = 40
PACK_SIZE = 15

# Audit cases that MUST appear in the sample regardless of their current class.
# Two are still confidential; one was flipped to public (Case 1).
# Including all three tests whether external labelers agree with our decisions.
AUDIT_CASES = [
    "wildchat-3aba9fbe9541a6262ea606a7af9fd328",  # Case 1: Vue/SCSS Chinese placeholder; we flipped to public
    "wildchat-d5a328a8e0a5c8e7215ea2bd01ad8eff",  # Case 2: Canadian work permit; kept confidential
    "wildchat-1b26ab6746ab5e4178abe77c22858085",  # Case 3: Russian mental health; kept confidential
]


def load_prompt_pool() -> dict[str, str]:
    """Read the master prompts_pool.jsonl (5k rows, broadest coverage)."""
    pool: dict[str, str] = {}
    for line in POOL_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        pool[row["id"]] = row["text"]
    return pool


def load_labels() -> dict[str, dict]:
    overrides: dict[str, str] = {}
    if OVERRIDES_FILE.exists():
        for line in OVERRIDES_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            overrides[o["id"]] = o["privacy"]

    labels: dict[str, dict] = {}
    for chunk in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        for line in chunk.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["id"] in overrides:
                row["privacy"] = overrides[row["id"]]
                row["overridden"] = True
            labels[row["id"]] = row
    return labels


def stratified_sample(
    labels: dict[str, dict],
    pool: dict[str, str],
    rng: random.Random,
) -> list[str]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for rid, lbl in labels.items():
        if rid not in pool:
            continue
        by_class[lbl["privacy"]].append(rid)

    print(
        f"  class sizes: public={len(by_class.get('public', []))}, "
        f"internal={len(by_class.get('internal', []))}, "
        f"confidential={len(by_class.get('confidential', []))}"
    )

    forced_by_class: dict[str, set[str]] = defaultdict(set)
    for case_id in AUDIT_CASES:
        if case_id in labels and case_id in pool:
            forced_by_class[labels[case_id]["privacy"]].add(case_id)

    def draw(class_name: str, target: int) -> list[str]:
        forced = forced_by_class.get(class_name, set())
        population = [rid for rid in by_class[class_name] if rid not in forced]
        need = target - len(forced)
        drawn = rng.sample(population, min(need, len(population)))
        return sorted(forced) + drawn

    n_conf_available = len(by_class.get("confidential", []))
    picked_conf = draw("confidential", n_conf_available)  # take all available
    picked_internal = draw("internal", N_INTERNAL)
    picked_public = draw("public", N_PUBLIC)

    combined = picked_conf + picked_internal + picked_public
    rng.shuffle(combined)
    return combined


def build_rubric() -> str:
    """Rubric shown to each external labeler. Derived from SYSTEM_PROMPT in
    scripts/label_wildchat.py verbatim, with a short format reminder appended."""
    return """\
# Privacy Classification Rubric (frozen)

You are labeling user prompts for an enterprise AI routing system. For each
prompt, output exactly three labels: **domain**, **complexity**, **privacy**.

## TAXONOMY

### domain — what TYPE of task (not topic):

| domain          | meaning                                                        |
|-----------------|----------------------------------------------------------------|
| chat            | conversational, open-ended, no clear deliverable               |
| code            | writing/debugging/explaining code; shell commands; SQL         |
| reasoning       | logic, math proofs, step-by-step analysis, planning            |
| extraction      | pulling structured data from unstructured input                |
| classification  | assigning labels/categories to input                           |
| summarization   | condensing longer input into shorter output                    |
| creative        | fiction, poetry, brainstorming, roleplay, marketing copy       |

### complexity — cognitive load for a correct answer:

| complexity | meaning                                                              |
|------------|----------------------------------------------------------------------|
| simple     | one-step lookup/answer; trivially verifiable                         |
| moderate   | multi-step but bounded scope                                         |
| complex    | architecture, system design, advanced reasoning                      |
| critical   | medical diagnosis, legal advice, financial planning, compliance —    |
|            | wrong answer has material real-world consequences                    |

### privacy — sensitivity of the CONTENT itself:

| privacy       | meaning                                                           |
|---------------|-------------------------------------------------------------------|
| public        | no sensitive info; could be posted on a public forum              |
| internal      | business content, work tasks, routine questions                   |
| confidential  | contains PII (SSN, credit cards, real names+context),             |
|               | secrets (API keys, passwords, tokens),                            |
|               | medical/legal/financial specifics tied to a person or org         |

## RULES (important)

1. **Classify the REQUEST, not the topic mentioned.**
   "Summarize this Python tutorial" → domain=summarization, not code.

2. **Asymmetric safety.** When privacy is ambiguous between `internal` and
   `confidential`, choose `confidential`. False negatives are compliance
   incidents; overclassification is safe.

3. **Never output `public` if the message contains real names, addresses,
   phone numbers, emails, account numbers, or any identifier.**

4. **Placeholder patterns are NOT confidential.** Text like `BOT_TOKEN`,
   `YOUR_API_KEY_HERE`, `user@example.com`, `张三` (Chinese literal for
   "John Doe"), `13845257654` (standard Chinese tutorial phone placeholder)
   are template placeholders, not real data. Classify the prompt by what
   the user is actually asking, not by the presence of placeholder-shaped
   strings.

5. **Fictional-character medical/legal narratives are NOT confidential.**
   A story about "Patricia Bertier's cardiac arrhythmia" is creative
   writing, not PHI. A real person's symptoms ("I have depression and
   anxiety, please help") IS confidential.

## OUTPUT FORMAT

For each prompt you see, output exactly one JSON line. Do not add preamble,
explanation, or trailing text — only the JSON lines.

```
{"id": "wildchat-xxxxxx", "domain": "...", "complexity": "...", "privacy": "..."}
```

The `id` must match the id shown above each prompt.
"""


def build_pack(
    pack_num: int,
    total_packs: int,
    rows: list[tuple[str, str]],
    rubric: str,
) -> str:
    header = f"""\
# IRR Labeling Pack — Batch {pack_num} of {total_packs}

You are helping label {len(rows)} prompts in this batch. Read the rubric
carefully, then output one JSON line per prompt, matching the id shown.

---

{rubric}

---

## Prompts to classify ({len(rows)} total)

"""
    body_parts = [header]
    for idx, (rid, text) in enumerate(rows, start=1):
        body_parts.append(f"### Prompt {idx} — id: `{rid}`\n\n")
        body_parts.append("```text\n")
        body_parts.append(text.rstrip())
        body_parts.append("\n```\n\n")

    footer = f"""\
---

## REMINDER — Output format

Now output exactly {len(rows)} JSON lines, one per prompt, in order, using the
ids shown above. Output ONLY the JSON lines — no preamble, no explanation,
no trailing text.

```
{{"id": "wildchat-...", "domain": "...", "complexity": "...", "privacy": "..."}}
... ({len(rows)} lines total)
```
"""
    body_parts.append(footer)
    return "".join(body_parts)


def build_readme(n_packs: int, n_total: int) -> str:
    return f"""\
# IRR Labeling Workflow

## What this is

You are collecting independent privacy-classification labels from **three**
AI models (Claude, GPT, Gemini) on the same {n_total} prompts. Comparing their
agreement gives an **inter-rater reliability** (IRR) number we can publish.

This addresses the single-labeler credibility gap in the research.

## What you'll do

For each of the {n_packs} pack files in this directory, paste the file into
GPT (via Copilot) AND into Gemini, save each response, and tell me when done.

Estimated time: 45-60 minutes total.

---

## STEP-BY-STEP WORKFLOW

### Before you start

Create a folder to hold responses:

```
tests/classification/irr/responses/
```

Windows: right-click, New Folder, name it `responses`.

### For each pack (`pack_01.md` through `pack_{n_packs:02d}.md`)

Repeat this loop {n_packs} times, once per pack.

#### A. GPT (via Microsoft Copilot) — "Think Deeper" mode

1. Open **copilot.microsoft.com** (or the Copilot app).
2. In the chat input area, click **"Think Deeper"** so the deep-thinking mode
   is ON (the button highlights). If you don't see it, use the standard mode;
   deep-thinking just improves accuracy.
3. **Important: start a NEW chat** (click "New chat" in sidebar) so no prior
   context leaks in. IRR requires independent labeling.
4. Open `pack_NN.md` in a text editor (or GitHub / VS Code).
5. Select ALL (Ctrl+A), COPY (Ctrl+C).
6. Paste into Copilot's input box. Press Enter.
7. Wait for the response. Deep-thinking may take 30-90 seconds.
8. Copy the response (only the JSON lines — the lines that start with `{{`).
9. Save to `responses/pack_NN_gpt.jsonl` as plain UTF-8 text.
   **Important:** file must end with `.jsonl` (not `.txt`).

#### B. Gemini — deep-thinking mode

1. Open **gemini.google.com**.
2. In the model selector (top-left), choose **"Gemini 2.5 Pro"** (or the
   highest-capability "thinking" / "deep thinking" model available).
3. **Start a NEW chat** (click "New chat").
4. Same process: select-all from `pack_NN.md`, paste, send.
5. Wait for response.
6. Copy just the JSON lines.
7. Save to `responses/pack_NN_gemini.jsonl`.

#### C. Check you have both files

After each pack, confirm you have:
- `responses/pack_NN_gpt.jsonl`
- `responses/pack_NN_gemini.jsonl`

Both files should contain roughly 15 JSON lines (one per prompt in that pack).

### When all {n_packs} packs are done

Tell me "IRR labeling done" and I'll run the scoring script. I'll produce:
- Fleiss' κ (multi-rater agreement)
- Pairwise Cohen's κ (Claude-vs-GPT, Claude-vs-Gemini, GPT-vs-Gemini)
- Confusion matrix per pair
- Disagreement report (rows where we disagree, for adjudication)
- Structural-miss verdict: did GPT/Gemini catch the 3 audit cases?

---

## TROUBLESHOOTING

### The model's response is truncated
Reply to the model: **"Please continue from prompt N"** (where N is the last
you saw). Paste additional lines into the same response file.

### The model refuses (safety filter on sensitive content)
Some prompts contain sensitive content (real credentials, medical text,
immigration text) — this is intentional and necessary for the study. If a
model refuses:
- Try the other model (Gemini may accept what Copilot refuses, or vice versa)
- Note which pack+prompt was refused in a text file `responses/refusals.txt`
- Skip that prompt and continue with the rest of the pack
- A refusal is itself a data point (it means that model can't help here)

### The model adds preamble or explanation
That's fine. When saving, only keep the JSON lines — lines starting with `{{`.
Delete everything else before saving.

### The format is weird (wrong keys, missing fields)
Paste the whole response to me in our chat and I'll reformat it. Don't worry
about fixing it yourself.

### A file got corrupted or you want to redo
Just redo that pack. Overwrite the response file. No harm done.

---

## NOTES ON SAFETY

- These prompts were drawn from the public **WildChat** dataset. All data is
  already public. But some prompts contain real-looking credentials and
  personal disclosures (that's what makes them `confidential`-class).
- Use only the in-chat models (Copilot, Gemini). Do NOT upload the pack
  files to any permanent document-storage service.
- When done, you can delete the pack and response files; I only need the
  scoring output.
"""


def main() -> None:
    if not POOL_FILE.exists():
        raise SystemExit(f"Missing pool file: {POOL_FILE}")
    if not CHUNKS_DIR.exists():
        raise SystemExit(f"Missing chunks dir: {CHUNKS_DIR}")

    IRR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading pool from {POOL_FILE.name}...")
    pool = load_prompt_pool()
    print(f"  {len(pool)} pool rows")

    print("Loading labels (chunks + overrides)...")
    labels = load_labels()
    print(f"  {len(labels)} labeled rows")

    rng = random.Random(SEED)
    ids = stratified_sample(labels, pool, rng)
    print(f"Sampled {len(ids)} rows total")
    for case_id in AUDIT_CASES:
        if case_id in ids:
            print(f"  audit-case included: {case_id} (class={labels[case_id]['privacy']})")
        else:
            print(f"  WARNING: audit-case MISSING: {case_id}")

    rubric = build_rubric()
    (IRR_DIR / "rubric.md").write_text(rubric, encoding="utf-8")
    print(f"Wrote {IRR_DIR / 'rubric.md'}")

    # Distribute rows across packs of ~PACK_SIZE. The last pack absorbs the overflow
    # (can be slightly larger or smaller). 10 packs target.
    full_packs = len(ids) // PACK_SIZE
    remainder = len(ids) - full_packs * PACK_SIZE
    if remainder == 0:
        pack_counts = [PACK_SIZE] * full_packs
    elif remainder <= PACK_SIZE // 3:
        # Small remainder: absorb into last full pack rather than create a tiny pack.
        pack_counts = [PACK_SIZE] * (full_packs - 1) + [PACK_SIZE + remainder]
    else:
        pack_counts = [PACK_SIZE] * full_packs + [remainder]
    n_packs_actual = len(pack_counts)
    assert sum(pack_counts) == len(ids), (pack_counts, len(ids))

    manifest_lines: list[str] = []
    truth_lines: list[str] = []
    cursor = 0
    for pack_idx in range(n_packs_actual):
        count = pack_counts[pack_idx]
        pack_ids = ids[cursor:cursor + count]
        cursor += count
        rows = [(rid, pool[rid]) for rid in pack_ids]
        pack_md = build_pack(pack_idx + 1, n_packs_actual, rows, rubric)
        pack_path = IRR_DIR / f"pack_{pack_idx + 1:02d}.md"
        pack_path.write_text(pack_md, encoding="utf-8")
        print(f"Wrote {pack_path.name} ({len(pack_ids)} rows)")

        for rid in pack_ids:
            manifest_lines.append(json.dumps({
                "id": rid,
                "pack": pack_idx + 1,
            }))
            lbl = labels[rid]
            truth_lines.append(json.dumps({
                "id": rid,
                "domain": lbl["domain"],
                "complexity": lbl["complexity"],
                "privacy": lbl["privacy"],
                "overridden": lbl.get("overridden", False),
            }))

    (IRR_DIR / "sample_manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (IRR_DIR / "truth_keys.jsonl").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")
    print(f"Wrote {IRR_DIR / 'sample_manifest.jsonl'}")
    print(f"Wrote {IRR_DIR / 'truth_keys.jsonl'}")

    (IRR_DIR / "README.md").write_text(build_readme(n_packs_actual, len(ids)), encoding="utf-8")
    print(f"Wrote {IRR_DIR / 'README.md'}")

    print("\nDone.  Next step: user follows tests/classification/irr/README.md")


if __name__ == "__main__":
    main()
