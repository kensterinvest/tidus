# Contributing to Tidus

Thank you for your interest in contributing! This document covers how to set up your development environment, submit changes, and follow project conventions.

## Development Setup

```bash
git clone https://github.com/lapkei01/tidus.git
cd tidus
uv sync --all-extras
cp .env.example .env
# Add at least OLLAMA_BASE_URL for local testing without API keys
```

Run the tests:
```bash
pytest
```

Run the dev server:
```bash
uvicorn tidus.main:app --reload
```

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, always deployable. Tagged on each release. |
| `develop` | Integration branch. All feature branches merge here first. |
| `phase/N-name` | Phase implementation branches (core team). |
| `feat/short-description` | Community feature branches. |
| `fix/short-description` | Bug fix branches. |

**Workflow:** `feat/my-feature` → PR → `develop` → PR → `main`

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Qwen adapter
fix: correct token count for Mistral sentencepiece
docs: update ROI calculator with Gemini Flash pricing
refactor: extract fallback logic into selector helper
test: add unit tests for budget enforcer rolling_30d period
chore: bump anthropic SDK to 0.32.0
```

## Pull Request Checklist

Before submitting a PR:
- [ ] `pytest` passes with no failures
- [ ] New code has docstrings on public functions/classes
- [ ] New adapters follow the `AbstractModelAdapter` interface in `tidus/adapters/base.py`
- [ ] New models are added to `config/models.yaml` with all required fields
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`
- [ ] PR description explains *why*, not just *what*

## Code Style

We use `ruff` for linting and formatting:
```bash
ruff check tidus/ tests/
ruff format tidus/ tests/
```

Line length: 100 characters. Python 3.12+ type hints required on all public APIs.

## Adding a New Vendor Adapter

1. Create `tidus/adapters/{vendor}_adapter.py`
2. Implement `AbstractModelAdapter` — complete, stream_complete, health_check, count_tokens
3. Decorate with `@register_adapter`
4. Add model entries to `config/models.yaml`
5. Add `{VENDOR}_API_KEY` to `.env.example`
6. Add integration tests in `tests/adapters/`
7. Update `docs/adapters.md`

## Developer Certificate of Origin (DCO)

By contributing, you certify that your contribution was written by you and you have the right to submit it under the Apache 2.0 license.

Sign your commits:
```bash
git commit -s -m "feat: add Qwen adapter"
```

## Questions?

Open a GitHub issue or email lapkei01@gmail.com.
