from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and return raw YAML content from a file path.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file is not valid YAML.

    Example:
        data = load_yaml("config/models.yaml")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {p}: {exc}") from exc


def load_yaml_as(path: str | Path, model: type[BaseModel]) -> BaseModel:
    """Load YAML and parse into a Pydantic model.

    Raises ValidationError if the YAML does not match the model schema.

    Example:
        config = load_yaml_as("config/policies.yaml", PoliciesConfig)
    """
    data = load_yaml(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Config validation failed for {path}:\n{exc}") from exc


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary back to a YAML file (used by price_sync).

    Example:
        dump_yaml({"models": [...]}, "config/models.yaml")
    """
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
