"""MCP tool definitions for Tidus.

Tools exposed to MCP clients:
    tidus_route_task      — select the best model without executing (returns model_id + cost)
    tidus_complete_task   — route + execute via the chosen adapter (returns content)
    tidus_get_budget_status — check team spend vs limit
    tidus_list_models     — list all models with their tier / enabled status
"""

from __future__ import annotations

TOOLS = [
    {
        "name": "tidus_route_task",
        "description": (
            "Select the cheapest capable model for a task without executing it. "
            "Returns the chosen model_id, vendor, estimated cost, and rejection reasons "
            "if no model is suitable."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["team_id", "complexity", "domain", "estimated_input_tokens", "messages"],
            "properties": {
                "team_id": {"type": "string", "description": "Team or workflow identifier"},
                "complexity": {
                    "type": "string",
                    "enum": ["simple", "moderate", "complex", "critical"],
                    "description": "Task complexity level",
                },
                "domain": {
                    "type": "string",
                    "enum": ["chat", "code", "reasoning", "extraction",
                             "classification", "summarization", "creative"],
                    "description": "Task domain",
                },
                "estimated_input_tokens": {
                    "type": "integer",
                    "description": "Approximate number of input tokens",
                },
                "messages": {
                    "type": "array",
                    "description": "OpenAI-format messages array",
                    "items": {"type": "object"},
                },
                "privacy": {
                    "type": "string",
                    "enum": ["public", "internal", "confidential"],
                    "default": "public",
                },
                "max_cost_usd": {
                    "type": "number",
                    "description": "Maximum acceptable cost in USD (optional)",
                },
            },
        },
    },
    {
        "name": "tidus_complete_task",
        "description": (
            "Route and execute an AI task. Selects the cheapest capable model, "
            "calls the vendor adapter, deducts budget, and returns the response content."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["team_id", "complexity", "domain", "estimated_input_tokens", "messages"],
            "properties": {
                "team_id": {"type": "string"},
                "complexity": {
                    "type": "string",
                    "enum": ["simple", "moderate", "complex", "critical"],
                },
                "domain": {
                    "type": "string",
                    "enum": ["chat", "code", "reasoning", "extraction",
                             "classification", "summarization", "creative"],
                },
                "estimated_input_tokens": {"type": "integer"},
                "messages": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "privacy": {
                    "type": "string",
                    "enum": ["public", "internal", "confidential"],
                    "default": "public",
                },
                "max_cost_usd": {"type": "number"},
                "estimated_output_tokens": {"type": "integer", "default": 256},
                "agent_depth": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "tidus_get_budget_status",
        "description": "Check the current spend vs. limit for a team.",
        "inputSchema": {
            "type": "object",
            "required": ["team_id"],
            "properties": {
                "team_id": {"type": "string"},
            },
        },
    },
    {
        "name": "tidus_list_models",
        "description": "List all models in the Tidus registry with tier, enabled status, and pricing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "enabled_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, only return enabled models",
                },
            },
        },
    },
]
