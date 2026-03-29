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
    redis_url: str | None = None

    # ── Config Paths ─────────────────────────────────────────────────────────
    models_config_path: str = "config/models.yaml"
    budgets_config_path: str = "config/budgets.yaml"
    policies_config_path: str = "config/policies.yaml"

    # ── Service ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "development"

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


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
