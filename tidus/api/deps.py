"""Shared FastAPI dependencies — singletons initialized once at app startup.

Call `build_singletons()` from the lifespan context manager, then inject
components into route handlers via FastAPI's `Depends()` system.

Example:
    @router.post("/route")
    async def route(selector: Annotated[ModelSelector, Depends(get_selector)]):
        ...
"""

from __future__ import annotations

from tidus.audit.logger import AuditLogger
from tidus.budget.enforcer import BudgetEnforcer
from tidus.budget.policies import load_budget_policies
from tidus.cache.exact_cache import ExactCache
from tidus.cost.counter import RedisSpendCounter, SpendCounter
from tidus.cost.engine import CostEngine
from tidus.cost.logger import CostLogger
from tidus.guardrails.agent_guard import AgentGuard
from tidus.guardrails.session_store import SessionStore
from tidus.metering.service import MeteringService
from tidus.models.guardrails import GuardrailPolicy
from tidus.registry.effective_registry import EffectiveRegistry
from tidus.registry.override_manager import OverrideManager
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.selector import ModelSelector
from tidus.settings import get_settings
from tidus.utils.yaml_loader import load_yaml

# ── Module-level singletons ───────────────────────────────────────────────────

_registry: EffectiveRegistry | None = None
_selector: ModelSelector | None = None
_enforcer: BudgetEnforcer | None = None
_guardrail_policy: GuardrailPolicy | None = None
_session_store: SessionStore | None = None
_agent_guard: AgentGuard | None = None
_cost_logger: CostLogger | None = None
_audit_logger: AuditLogger | None = None
_metering: MeteringService | None = None
_override_manager: OverrideManager | None = None
_exact_cache: ExactCache | None = None


def _build_spend_counter(settings) -> SpendCounter | RedisSpendCounter:
    """Pick the SpendCounter backend based on settings.redis_url.

    Returns a Redis-backed counter when ``settings.redis_url`` is configured
    (required for multi-worker deployments so budget state is shared across
    processes); otherwise returns the in-memory counter.
    """
    if not settings.redis_url:
        return SpendCounter()
    from redis.asyncio import Redis
    client = Redis.from_url(settings.redis_url, decode_responses=False)
    return RedisSpendCounter(client, prefix=settings.redis_spend_counter_prefix)


async def build_singletons() -> None:
    """Initialize all shared singletons from config files.

    Async because EffectiveRegistry.build() queries the DB for the active
    revision. Called once from the FastAPI lifespan on startup. Safe to call
    again (re-initializes, useful for testing overrides).
    """
    global _registry, _selector, _enforcer, _guardrail_policy, _session_store, _agent_guard, _cost_logger, _audit_logger, _metering, _override_manager, _exact_cache

    settings = get_settings()

    from tidus.db.engine import get_session_factory as _get_sf
    sf = _get_sf()

    raw_policies = load_yaml(settings.policies_config_path)
    _guardrail_policy = GuardrailPolicy.model_validate(raw_policies["guardrails"])
    buffer_pct = raw_policies["cost"]["estimate_buffer_pct"]

    # Build the layered registry: DB revision + overrides + telemetry.
    # Falls back to YAML load if no active revision exists (e.g. test environments).
    _registry = await EffectiveRegistry.build(sf, settings.models_config_path)

    budgets = load_budget_policies(settings.budgets_config_path)
    counter = _build_spend_counter(settings)
    _enforcer = BudgetEnforcer(budgets, counter)

    matcher = CapabilityMatcher(_guardrail_policy)
    engine = CostEngine(buffer_pct=buffer_pct)
    _selector = ModelSelector(_registry, _enforcer, matcher, engine)

    _session_store = SessionStore()
    _agent_guard = AgentGuard(_guardrail_policy, _session_store)

    _cost_logger = CostLogger(sf)
    _audit_logger = AuditLogger(sf)
    _metering = MeteringService(sf)
    _override_manager = OverrideManager(sf, audit_logger=_audit_logger)

    if settings.cache_enabled:
        _exact_cache = ExactCache(
            ttl_seconds=settings.cache_ttl_seconds,
            max_size=settings.cache_max_size,
        )
    else:
        _exact_cache = None


# ── Dependency getters (used with FastAPI Depends) ────────────────────────────

def get_registry() -> EffectiveRegistry:
    assert _registry is not None, "Singletons not built — call build_singletons() at startup"
    return _registry


def get_selector() -> ModelSelector:
    assert _selector is not None, "Singletons not built — call build_singletons() at startup"
    return _selector


def get_enforcer() -> BudgetEnforcer:
    assert _enforcer is not None, "Singletons not built — call build_singletons() at startup"
    return _enforcer


def get_guardrail_policy() -> GuardrailPolicy:
    assert _guardrail_policy is not None, "Singletons not built — call build_singletons() at startup"
    return _guardrail_policy


def get_session_store() -> SessionStore:
    assert _session_store is not None, "Singletons not built — call build_singletons() at startup"
    return _session_store


def get_agent_guard() -> AgentGuard:
    assert _agent_guard is not None, "Singletons not built — call build_singletons() at startup"
    return _agent_guard


def get_cost_logger() -> CostLogger:
    assert _cost_logger is not None, "Singletons not built — call build_singletons() at startup"
    return _cost_logger


def get_audit_logger() -> AuditLogger:
    assert _audit_logger is not None, "Singletons not built — call build_singletons() at startup"
    return _audit_logger


def get_metering() -> MeteringService:
    assert _metering is not None, "Singletons not built — call build_singletons() at startup"
    return _metering


def get_override_manager() -> OverrideManager:
    assert _override_manager is not None, "Singletons not built — call build_singletons() at startup"
    return _override_manager


def get_exact_cache() -> ExactCache | None:
    """Return the process-wide ExactCache, or None if cache is disabled."""
    return _exact_cache


def get_session_factory():
    from tidus.db.engine import get_session_factory as _get_sf
    return _get_sf()
