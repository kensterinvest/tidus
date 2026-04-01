from collections.abc import AsyncGenerator

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, JSON, String, Text, func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from tidus.settings import get_settings


class Base(DeclarativeBase):
    pass


# ── ORM Tables ────────────────────────────────────────────────────────────────

class CostRecordORM(Base):
    __tablename__ = "cost_records"

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, index=True)
    team_id = Column(String, nullable=False, index=True)
    workflow_id = Column(String, nullable=True, index=True)
    agent_session_id = Column(String, nullable=True)
    agent_depth = Column(Integer, default=0)
    routing_decision_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False, index=True)
    vendor = Column(String, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    fallback_used = Column(Boolean, default=False)
    fallback_from = Column(String, nullable=True)


class BudgetPolicyORM(Base):
    __tablename__ = "budget_policies"

    policy_id = Column(String, primary_key=True)
    scope = Column(String, nullable=False)       # "team" | "workflow"
    scope_id = Column(String, nullable=False, index=True)
    period = Column(String, nullable=False)
    limit_usd = Column(Float, nullable=False)
    warn_at_pct = Column(Float, default=0.80)
    hard_stop = Column(Boolean, default=True)


class PriceChangeLogORM(Base):
    __tablename__ = "price_change_log"

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False, index=True)
    vendor = Column(String, nullable=False)
    field_changed = Column(String, nullable=False)   # input_price | output_price | max_context
    old_value = Column(Float, nullable=False)
    new_value = Column(Float, nullable=False)
    delta_pct = Column(Float, nullable=False)
    detected_at = Column(DateTime, server_default=func.now(), nullable=False)
    source = Column(String, default="weekly_sync")


class RoutingDecisionORM(Base):
    __tablename__ = "routing_decisions"

    decision_id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, index=True)
    team_id = Column(String, nullable=False, index=True)
    selected_model_id = Column(String, nullable=True)
    selected_vendor = Column(String, nullable=True)
    rejection_reason = Column(String, nullable=True)
    explanation = Column(Text, nullable=False)
    estimated_cost_usd = Column(Float, nullable=True)
    fallback_from = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False, index=True)


class AuditLogORM(Base):
    """Tamper-evident audit log for SOC 2 / ISO 27001 / HIPAA compliance.

    Records who (actor_team_id + actor_role + actor_sub) did what (action)
    to which resource (resource_type + resource_id), and whether it succeeded.
    """

    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    actor_team_id = Column(String, nullable=False, index=True)
    actor_role = Column(String, nullable=False)
    actor_sub = Column(String, nullable=False)           # JWT sub / "dev" in dev-mode
    action = Column(String, nullable=False, index=True)  # e.g. "route", "complete", "budget.create"
    resource_type = Column(String, nullable=True)        # e.g. "task", "budget_policy"
    resource_id = Column(String, nullable=True, index=True)
    outcome = Column(String, nullable=False)             # "success" | "rejected" | "error"
    rejection_reason = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)  # arbitrary extra context


# ── Engine & Session Factory ──────────────────────────────────────────────────

_engine = None
_session_factory: async_sessionmaker | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        is_sqlite = "sqlite" in settings.database_url
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        pool_kwargs: dict = {}
        if not is_sqlite:
            # PostgreSQL production pool settings
            pool_kwargs = {
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
            }
        _engine = create_async_engine(
            settings.database_url,
            connect_args=connect_args,
            echo=settings.environment == "development",
            **pool_kwargs,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async DB session."""
    async with get_session_factory()() as session:
        yield session


async def create_tables() -> None:
    """Create all tables. Called at app startup."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
