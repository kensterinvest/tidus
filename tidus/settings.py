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


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
