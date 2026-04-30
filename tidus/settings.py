from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Vendor API Keys ──────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    mistral_api_key: str = ""
    deepseek_api_key: str = ""
    xai_api_key: str = ""
    moonshot_api_key: str = ""

    # ── Local Models ─────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./tidus.db"

    # ── Redis (optional) ─────────────────────────────────────────────────────
    # When set, SpendCounter uses Redis INCRBYFLOAT + a Lua script for atomic
    # cross-worker budget enforcement. Leave unset to use in-memory counters
    # (safe for single-worker deployments and tests).
    redis_url: str | None = None
    redis_spend_counter_prefix: str = "tidus:spend"

    # ── Config Paths ─────────────────────────────────────────────────────────
    models_config_path: str = "config/models.yaml"
    budgets_config_path: str = "config/budgets.yaml"
    policies_config_path: str = "config/policies.yaml"

    # ── Service ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "production"
    # Explicit CORS origin allowlist. Use "*" only in development.
    # Example: "https://app.example.com,https://admin.example.com"
    cors_allowed_origins: str = ""

    # ── Tidus Licensing ──────────────────────────────────────────────────────
    tidus_license_key: str = ""
    tidus_tier: str = "community"  # community | pro | business | enterprise

    # ── OIDC / SSO ───────────────────────────────────────────────────────────
    # Leave oidc_issuer_url empty to run in dev mode (no auth required).
    # Supported providers: Okta, Azure Entra ID, Google Workspace, Auth0, generic OIDC.
    oidc_issuer_url: str = ""          # e.g. https://my.okta.com/oauth2/default
    oidc_client_id: str = ""           # JWT audience claim value
    oidc_client_secret: str = ""       # For token introspection if needed (optional)
    oidc_team_claim: str = "tid"       # JWT claim holding the team_id
    oidc_role_claim: str = "role"      # JWT claim holding the Tidus role
    # Stage B telemetry uses tenant_id (broader than team) for per-tenant
    # fine-tuning hooks. Resolution order: JWT claim → X-Tenant-ID header →
    # fall back to team_id. In deployments where teams ARE the tenants,
    # leaving the claim absent just defaults to team_id — no config change needed.
    oidc_tenant_claim: str = "tenant_id"
    tenant_header_name: str = "X-Tenant-ID"

    # Dev-mode fallback identity (used when oidc_issuer_url is unset)
    oidc_dev_team_id: str = "team-dev"
    oidc_dev_role: str = "admin"

    # ── Registry (v1.1.0) ────────────────────────────────────────────────────
    # Optional pricing feed URL. When set, TidusPricingFeedSource pulls price
    # data from this endpoint. No customer data is sent — only schema_version.
    # Leave empty to use the built-in hardcoded price table only.
    tidus_pricing_feed_url: str = ""

    # How long to retain SUPERSEDED revisions before cleanup (days).
    registry_revision_retention_days: int = 90

    # HMAC-SHA256 signing key for the override export bundle.
    # When set, GET /api/v1/registry/overrides/export includes an
    # X-Tidus-Signature header. Leave empty to export unsigned (with a warning).
    tidus_registry_export_signing_key: str = ""

    # HMAC-SHA256 key for verifying pricing feed responses.
    # When set, feed responses without a valid X-Tidus-Signature are rejected.
    # Leave empty to accept unsigned responses (logs a warning).
    tidus_pricing_feed_signing_key: str = ""

    # Circuit breaker: open after this many consecutive feed failures.
    pricing_feed_failure_threshold: int = 5

    # Circuit breaker: seconds before transitioning OPEN → HALF-OPEN.
    pricing_feed_reset_timeout_seconds: int = 300

    # ── Vendor model discovery ──────────────────────────────────────────────
    # Auto-discovery polls each vendor's `/v1/models` endpoint to detect new
    # models. Discoveries are SURFACE-only (report + JSON sidecar) — never
    # auto-routed; promotion to the active routing catalog still requires a
    # human edit to `config/models.yaml` + `tidus/sync/pricing/hardcoded_source.py`.
    discovery_enabled: bool = True
    discovery_state_path: str = "reports/discovered_models.json"
    discovery_request_timeout_seconds: float = 15.0

    # ── Response Cache (v1.1 Pillar 3) ───────────────────────────────────────
    # Exact-match cache: hash(team_id + messages + model_id) → response.
    # Disabled in prod only for debugging / A-B testing cache impact.
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10_000

    # ── Adapter resilience (Fix 18) ──────────────────────────────────────────
    # Per-call timeout for vendor API requests. Beyond this wall-clock, the
    # request is cancelled and counted as a transient failure for retry.
    adapter_timeout_seconds: float = 60.0
    # Total attempts on transient failures (429/5xx/timeout). 3 = one retry.
    # Auth + client errors are never retried regardless of this setting.
    adapter_max_retries: int = 3
    adapter_base_delay_seconds: float = 0.5

    # ── Auto-classification layer (v1.3.0) ───────────────────────────────────
    # Master toggle. When False, callers MUST supply classification fields.
    auto_classify_enabled: bool = True

    # Recipe B encoder (frozen all-MiniLM-L6-v2 + sklearn LR heads).
    # Directory must contain: domain_head.joblib, complexity_head.joblib,
    # privacy_head.joblib, label_mappings.json
    classify_encoder_dir: str = "tidus/classification/weights_b"
    classify_encoder_max_chars: int = 1200

    # Tier 5 — Local LLM escalation (Enterprise SKU only, requires GPU).
    # When False (CPU-only SKU default), topic-bearing confidentials miss the
    # ~10.8% recall gap — see docs/hardware-requirements.md.
    classify_tier5_enabled: bool = False
    classify_tier5_model: str = "phi3.5:3.8b-mini-instruct-q4_K_M"
    classify_tier5_rate_limit_per_minute: int = 60  # per-worker

    # Per-head confidence gates for tier escalation (A.4 will consume these).
    classify_privacy_threshold: float = 0.75
    classify_domain_threshold: float = 0.70
    classify_complexity_threshold: float = 0.65

    # Minimum encoder confidence to accept a "public" privacy verdict.
    # Below this, the merge rule downgrades the label to "internal" so we
    # never ship a weakly-supported "public" (plan.md §What-NOT-to-do:
    # "Never default privacy to public"). Separate from the escalation
    # threshold above because these serve different decisions.
    classify_privacy_public_floor: float = 0.70

    # Tier 2b — Presidio NER (parallel to Tier 2 encoder).
    # Set parallel=False to demote to conditional Tier 3 (saves latency when
    # Presidio p95 > 30ms on deployment hardware). See plan.md §Tier 2b.
    classify_presidio_enabled: bool = True
    classify_presidio_parallel: bool = True
    # Maximum text length sent to Presidio (latency control on pathological
    # prompts). Beyond this, the tail is ignored for NER scoring.
    classify_presidio_max_chars: int = 5000
    # E1 (default) = PERSON alone triggers confidential (89.2% recall, 49% flag
    # rate per findings.md). E2 = PERSON requires encoder-non-public corroboration
    # (83.1% recall, 19% flag rate). Plan.md §Stage A ships E1 by default;
    # precision-preferred tenants can switch to E2.
    classify_presidio_rule: str = "E1"

    # Tier 5 cache (exact match on user message hash).
    classify_cache_ttl_seconds: int = 3600
    classify_cache_max_entries: int = 10_000

    # Stage B — PII-safe classification telemetry.
    # When enabled (default), the /classify endpoint emits a structured log
    # record per request with type-only features (entity types, regex-pattern
    # IDs, tier decided, classification axes, dim-reduced embedding, model
    # routed, latency). Never emits raw prompts. See plan.md §Stage B.
    classify_telemetry_enabled: bool = True
    # Path to the PCA 384->64 artifact (joblib). Produced by
    # scripts/fit_pca_64d.py. Relative paths resolved against the repo root.
    classify_pca_path: str = "tidus/classification/weights_b/pca_64d.joblib"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
