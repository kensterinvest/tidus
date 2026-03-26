"""Budget policy loader — reads budgets.yaml into typed BudgetPolicy objects.

Example:
    policies = load_budget_policies("config/budgets.yaml")
    team_policy = next((p for p in policies if p.scope_id == "team-engineering"), None)
"""

from pathlib import Path
from typing import Any

from tidus.models.budget import BudgetPolicy
from tidus.utils.yaml_loader import load_yaml


def load_budget_policies(path: str | Path = "config/budgets.yaml") -> list[BudgetPolicy]:
    """Load and validate all budget policies from a YAML file.

    Returns an empty list if the file has no 'budgets' key.
    Raises FileNotFoundError if the file is missing.
    Raises ValueError if any policy fails Pydantic validation.

    Example:
        policies = load_budget_policies("config/budgets.yaml")
    """
    raw: dict[str, Any] = load_yaml(path)
    entries: list[dict] = raw.get("budgets", [])
    return [BudgetPolicy.model_validate(entry) for entry in entries]
