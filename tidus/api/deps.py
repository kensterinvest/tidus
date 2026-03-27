"""Shared FastAPI dependencies — singletons initialized once at app startup.

Call `build_singletons()` from the lifespan context manager, then inject
components into route handlers via FastAPI's `Depends()` system.

Example:
    @router.post("/route")
    async def route(selector: Annotated[ModelSelector, Depends(get_selector)]):
        ...
"""

from __future__ import annotations

from tidus.budget.enforcer import BudgetEnforcer
from tidus.budget.policies import load_budget_policies
from tidus.cost.counter import SpendCounter
from tidus.cost.engine import CostEngine
from tidus.cost.logger import CostLogger
from tidus.guardrails.agent_guard import AgentGuard
from tidus.guardrails.session_store import SessionStore
from tidus.models.guardrails import GuardrailPolicy
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.registry import ModelRegistry
from tidus.router.selector import ModelSelector
from tidus.settings import get_settings
from tidus.utils.yaml_loader import load_yaml

# ── Module-level singletons ───────────────────────────────────────────────────

_registry: ModelRegistry | None = None
_selector: ModelSelector | None = None
_enforcer: BudgetEnforcer | None = None
_guardrail_policy: GuardrailPolicy | None = None
_session_store: SessionStore | None = None
_agent_guard: AgentGuard | None = None
_cost_logger: CostLogger | None = None


def build_singletons() -> None:
    """Initialize all shared singletons from config files.

    Called once from the FastAPI lifespan on startup. Safe to call again
    (re-initializes, useful for testing overrides).
    """
    global _registry, _selector, _enforcer, _guardrail_policy, _session_store, _agent_guard, _cost_logger

    settings = get_settings()

    raw_policies = load_yaml(settings.policies_config_path)
    _guardrail_policy = GuardrailPolicy.model_validate(raw_policies["guardrails"])
    buffer_pct = raw_policies["cost"]["estimate_buffer_pct"]

    _registry = ModelRegistry.load(settings.models_config_path)

    budgets = load_budget_policies(settings.budgets_config_path)
    counter = SpendCounter()
    _enforcer = BudgetEnforcer(budgets, counter)

    matcher = CapabilityMatcher(_guardrail_policy)
    engine = CostEngine(buffer_pct=buffer_pct)
    _selector = ModelSelector(_registry, _enforcer, matcher, engine)

    _session_store = SessionStore()
    _agent_guard = AgentGuard(_guardrail_policy, _session_store)

    from tidus.db.engine import get_session_factory
    _cost_logger = CostLogger(get_session_factory())


# ── Dependency getters (used with FastAPI Depends) ────────────────────────────

def get_registry() -> ModelRegistry:
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


def get_session_factory():
    from tidus.db.engine import get_session_factory as _get_sf
    return _get_sf()
