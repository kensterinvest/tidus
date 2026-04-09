"""Tidus MCP server — exposes Tidus as tools for AI agents.

Transport: stdio (local agents) or SSE (remote agents, future).
Tools: tidus_route_task, tidus_complete_task, tidus_get_budget_status, tidus_list_models.

Usage (stdio):
    tidus-mcp

Configure in Claude Desktop / Cursor:
    {
      "mcpServers": {
        "tidus": {
          "command": "tidus-mcp",
          "args": []
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json

import structlog

from tidus.mcp.tools import TOOLS
from tidus.utils.logging import configure_logging

log = structlog.get_logger(__name__)


async def _handle_route(args: dict) -> str:
    """Handle tidus_route_task — returns routing decision as JSON string."""
    from tidus.api.deps import build_singletons, get_registry, get_selector
    from tidus.db.engine import create_tables
    from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
    from tidus.router.selector import ModelSelectionError

    await create_tables()
    await build_singletons()
    selector = get_selector()
    registry = get_registry()

    task = TaskDescriptor(
        team_id=args["team_id"],
        complexity=Complexity(args["complexity"]),
        domain=Domain(args["domain"]),
        privacy=Privacy(args.get("privacy", "public")),
        estimated_input_tokens=args["estimated_input_tokens"],
        estimated_output_tokens=args.get("estimated_output_tokens", 256),
        messages=args["messages"],
        max_cost_usd=args.get("max_cost_usd"),
    )

    try:
        decision = await selector.select(task)
        spec = registry.get(decision.chosen_model_id)
        return json.dumps({
            "chosen_model_id": decision.chosen_model_id,
            "vendor": spec.vendor if spec else None,
            "estimated_cost_usd": decision.estimated_cost_usd,
            "tier": int(spec.tier) if spec else None,
            "score": decision.score,
        })
    except ModelSelectionError as exc:
        return json.dumps({
            "error": "no_model_selected",
            "stage": exc.stage,
            "rejections": [
                {"model_id": r.chosen_model_id, "reason": r.rejection_reason}
                for r in exc.rejections
            ],
        })


async def _handle_complete(args: dict) -> str:
    """Handle tidus_complete_task — routes + executes, returns content."""
    from tidus.adapters.adapter_factory import get_adapter
    from tidus.api.deps import build_singletons, get_enforcer, get_registry, get_selector
    from tidus.db.engine import create_tables
    from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
    from tidus.router.selector import ModelSelectionError

    await create_tables()
    await build_singletons()
    selector = get_selector()
    registry = get_registry()
    enforcer = get_enforcer()

    task = TaskDescriptor(
        team_id=args["team_id"],
        complexity=Complexity(args["complexity"]),
        domain=Domain(args["domain"]),
        privacy=Privacy(args.get("privacy", "public")),
        estimated_input_tokens=args["estimated_input_tokens"],
        estimated_output_tokens=args.get("estimated_output_tokens", 256),
        agent_depth=args.get("agent_depth", 0),
        messages=args["messages"],
        max_cost_usd=args.get("max_cost_usd"),
    )

    try:
        decision = await selector.select(task)
    except ModelSelectionError as exc:
        return json.dumps({
            "error": "no_model_selected",
            "stage": exc.stage,
        })

    spec = registry.get(decision.chosen_model_id)
    if spec is None:
        return json.dumps({"error": f"Model {decision.chosen_model_id!r} not found in registry"})

    try:
        adapter = get_adapter(spec.vendor)
    except KeyError:
        return json.dumps({"error": f"No adapter for vendor '{spec.vendor}'"})

    try:
        response = await adapter.complete(decision.chosen_model_id, task)
    except Exception as exc:
        return json.dumps({"error": f"Adapter error: {exc}"})

    actual_cost = (
        response.input_tokens / 1000 * spec.input_price
        + response.output_tokens / 1000 * spec.output_price
    )
    await enforcer.deduct(task.team_id, None, actual_cost)

    return json.dumps({
        "content": response.content,
        "model_id": response.model_id,
        "vendor": spec.vendor,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": actual_cost,
        "latency_ms": response.latency_ms,
    })


async def _handle_budget_status(args: dict) -> str:
    """Handle tidus_get_budget_status."""
    from tidus.api.deps import build_singletons, get_enforcer
    from tidus.db.engine import create_tables

    await create_tables()
    await build_singletons()
    enforcer = get_enforcer()
    team_id = args["team_id"]
    status = await enforcer.status(team_id=team_id)
    has_policy = status.policy_id != "none"
    return json.dumps({
        "team_id": team_id,
        "has_policy": has_policy,
        "spent_usd": status.spent_usd,
        "limit_usd": status.limit_usd if has_policy else None,
        "utilisation_pct": status.utilisation_pct if has_policy else None,
        "is_hard_stopped": status.is_hard_stopped,
    })


async def _handle_list_models(args: dict) -> str:
    """Handle tidus_list_models."""
    from tidus.api.deps import build_singletons, get_registry
    from tidus.db.engine import create_tables

    await create_tables()
    await build_singletons()
    registry = get_registry()

    enabled_only = args.get("enabled_only", False)
    specs = registry.list_enabled() if enabled_only else registry.list_all()
    return json.dumps([
        {
            "model_id": s.model_id,
            "vendor": s.vendor,
            "tier": int(s.tier),
            "enabled": s.enabled,
            "is_local": s.is_local,
            "input_price": s.input_price,
            "output_price": s.output_price,
            "max_context": s.max_context,
        }
        for s in specs
    ])


_HANDLERS = {
    "tidus_route_task": _handle_route,
    "tidus_complete_task": _handle_complete,
    "tidus_get_budget_status": _handle_budget_status,
    "tidus_list_models": _handle_list_models,
}


async def run_mcp_server() -> None:
    """Run the Tidus MCP server over stdio."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    configure_logging("INFO")
    server = Server("tidus")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        handler = _HANDLERS.get(name)
        if handler is None:
            result = json.dumps({"error": f"Unknown tool: {name!r}"})
        else:
            try:
                result = await handler(arguments)
            except Exception as exc:
                log.error("mcp_tool_error", tool=name, error=str(exc))
                result = json.dumps({"error": str(exc)})
        return [TextContent(type="text", text=result)]

    log.info("tidus_mcp_starting", transport="stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point: tidus-mcp"""
    asyncio.run(run_mcp_server())
