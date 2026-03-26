# Changelog

All notable changes to Tidus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: `pyproject.toml`, directory structure, git setup
- Full model registry `config/models.yaml` — 26 models across 8 vendor families (OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, xAI, Kimi, Ollama local)
- Budget configuration `config/budgets.yaml` with example team and workflow policies
- Routing & guardrail policies `config/policies.yaml`
- Pydantic v2 data models: `TaskDescriptor`, `ModelSpec`, `CostEstimate`, `CostRecord`, `BudgetPolicy`, `BudgetStatus`, `RoutingDecision`, `AgentSession`, `GuardrailPolicy`, `PriceChangeRecord`
- SQLAlchemy async ORM tables: `cost_records`, `budget_policies`, `price_change_log`, `routing_decisions`
- FastAPI app factory with lifespan (DB creation on startup)
- Structured JSON logging via `structlog`
- Safe YAML loader/writer utilities
- Open-source readiness files: `README.md`, `CONTRIBUTING.md`, `LICENSE` (Apache 2.0), `CODE_OF_CONDUCT.md`, `SECURITY.md`
- Documentation skeleton in `docs/`

---

<!-- Links will be filled in at first public release -->
[Unreleased]: https://github.com/lapkei01/tidus/compare/HEAD...HEAD
