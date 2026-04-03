"""Unit tests for the MCP server — tools, handlers, and protocol integration.

Tests run without vendor API keys by mocking the tokenizer and adapters.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mcp_server():
    """Build a fresh MCP Server with all handlers registered (same code path as server.py)."""
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    from tidus.mcp.tools import TOOLS

    server = Server("tidus-test")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        from tidus.mcp.server import _HANDLERS
        handler = _HANDLERS.get(name)
        if handler is None:
            result = json.dumps({"error": f"Unknown tool: {name!r}"})
        else:
            result = await handler(arguments)
        return [TextContent(type="text", text=result)]

    return server


def _call_handler(server):
    """Return the call_tool request handler from the server."""
    from mcp.types import CallToolRequest
    return server.request_handlers[CallToolRequest]


def _list_handler(server):
    """Return the list_tools request handler from the server."""
    from mcp.types import ListToolsRequest
    return server.request_handlers[ListToolsRequest]


async def _call(server, tool_name: str, arguments: dict) -> dict:
    """Call a tool and parse the JSON response."""
    from mcp.types import CallToolRequest, CallToolRequestParams
    handler = _call_handler(server)
    req = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=tool_name, arguments=arguments),
    )
    result = await handler(req)
    return json.loads(result.root.content[0].text)


# ── Tool registration ──────────────────────────────────────────────────────────

class TestToolRegistration:
    async def test_lists_four_tools(self, mcp_server):
        """Server exposes exactly 4 Tidus tools."""
        from mcp.types import ListToolsRequest
        handler = _list_handler(mcp_server)
        result = await handler(ListToolsRequest(method="tools/list"))
        tools = result.root.tools
        assert len(tools) == 4

    async def test_tool_names_are_correct(self, mcp_server):
        from mcp.types import ListToolsRequest
        handler = _list_handler(mcp_server)
        result = await handler(ListToolsRequest(method="tools/list"))
        names = {t.name for t in result.root.tools}
        assert names == {
            "tidus_route_task",
            "tidus_complete_task",
            "tidus_get_budget_status",
            "tidus_list_models",
        }

    async def test_each_tool_has_description_and_schema(self, mcp_server):
        from mcp.types import ListToolsRequest
        handler = _list_handler(mcp_server)
        result = await handler(ListToolsRequest(method="tools/list"))
        for tool in result.root.tools:
            assert tool.description, f"{tool.name} has no description"
            assert tool.inputSchema, f"{tool.name} has no inputSchema"
            assert "properties" in tool.inputSchema


# ── tidus_list_models ──────────────────────────────────────────────────────────

class TestListModels:
    async def test_returns_all_models(self, mcp_server):
        result = await _call(mcp_server, "tidus_list_models", {"enabled_only": False})
        assert isinstance(result, list)
        assert len(result) == 53

    async def test_each_model_has_required_fields(self, mcp_server):
        result = await _call(mcp_server, "tidus_list_models", {})
        for model in result:
            for field in ("model_id", "vendor", "tier", "enabled", "is_local",
                          "input_price", "output_price", "max_context"):
                assert field in model, f"Model {model.get('model_id')} missing field {field}"

    async def test_enabled_only_returns_subset(self, mcp_server):
        all_models = await _call(mcp_server, "tidus_list_models", {"enabled_only": False})
        enabled_only = await _call(mcp_server, "tidus_list_models", {"enabled_only": True})
        assert len(enabled_only) <= len(all_models)
        for m in enabled_only:
            assert m["enabled"] is True

    async def test_local_models_have_zero_price(self, mcp_server):
        result = await _call(mcp_server, "tidus_list_models", {})
        for m in result:
            if m["is_local"]:
                assert m["input_price"] == 0.0
                assert m["output_price"] == 0.0


# ── tidus_route_task ───────────────────────────────────────────────────────────

class TestRouteTask:
    async def test_simple_task_returns_decision(self, mcp_server):
        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=50)):
            result = await _call(mcp_server, "tidus_route_task", {
                "team_id": "test-team",
                "complexity": "simple",
                "domain": "chat",
                "estimated_input_tokens": 200,
                "messages": [{"role": "user", "content": "hello"}],
            })
        assert "chosen_model_id" in result
        assert "vendor" in result
        assert "tier" in result
        assert "estimated_cost_usd" in result
        assert "error" not in result

    async def test_simple_task_routes_to_cheap_tier(self, mcp_server):
        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=50)):
            result = await _call(mcp_server, "tidus_route_task", {
                "team_id": "test-team",
                "complexity": "simple",
                "domain": "chat",
                "estimated_input_tokens": 200,
                "messages": [{"role": "user", "content": "hello"}],
            })
        assert result["tier"] in (2, 3, 4), f"Simple task should not pick tier 1, got {result['tier']}"

    async def test_confidential_routes_to_local_model(self, mcp_server):
        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=50)):
            result = await _call(mcp_server, "tidus_route_task", {
                "team_id": "test-team",
                "complexity": "simple",
                "domain": "chat",
                "privacy": "confidential",
                "estimated_input_tokens": 200,
                "messages": [{"role": "user", "content": "secret data"}],
            })
        assert result["tier"] == 4, f"Confidential should route to tier 4 (local), got {result}"

    async def test_impossible_budget_returns_error(self, mcp_server):
        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=5000)):
            result = await _call(mcp_server, "tidus_route_task", {
                "team_id": "test-team",
                "complexity": "complex",
                "domain": "reasoning",
                "max_cost_usd": 0.000001,
                "estimated_input_tokens": 5000,
                "messages": [{"role": "user", "content": "hard problem"}],
            })
        assert "error" in result
        assert result["error"] == "no_model_selected"


# ── tidus_get_budget_status ────────────────────────────────────────────────────

class TestBudgetStatus:
    async def test_team_with_policy_returns_status(self, mcp_server):
        result = await _call(mcp_server, "tidus_get_budget_status", {"team_id": "team-engineering"})
        assert result["team_id"] == "team-engineering"
        assert result["has_policy"] is True
        assert result["limit_usd"] is not None
        assert result["spent_usd"] >= 0.0

    async def test_team_without_policy_has_no_policy_flag(self, mcp_server):
        result = await _call(mcp_server, "tidus_get_budget_status", {"team_id": "nonexistent-team"})
        assert result["team_id"] == "nonexistent-team"
        assert result["has_policy"] is False
        assert result["limit_usd"] is None


# ── tidus_complete_task ────────────────────────────────────────────────────────

class TestCompleteTask:
    async def test_complete_returns_content(self, mcp_server):
        from tidus.adapters.base import AdapterResponse
        mock_response = AdapterResponse(
            model_id="llama4-scout-ollama",
            content="MCP response content",
            input_tokens=15,
            output_tokens=10,
            latency_ms=200.0,
        )
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(return_value=mock_response)

        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=50)), \
             patch("tidus.adapters.adapter_factory.get_adapter", return_value=mock_adapter):
            result = await _call(mcp_server, "tidus_complete_task", {
                "team_id": "test-team",
                "complexity": "simple",
                "domain": "chat",
                "estimated_input_tokens": 200,
                "messages": [{"role": "user", "content": "hello"}],
            })

        assert "content" in result
        assert result["content"] == "MCP response content"
        assert "model_id" in result
        assert "cost_usd" in result
        assert result["cost_usd"] >= 0.0

    async def test_complete_has_all_response_fields(self, mcp_server):
        from tidus.adapters.base import AdapterResponse
        mock_response = AdapterResponse(
            model_id="llama4-scout-ollama",
            content="hello",
            input_tokens=5,
            output_tokens=3,
            latency_ms=100.0,
        )
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(return_value=mock_response)

        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=10)), \
             patch("tidus.adapters.adapter_factory.get_adapter", return_value=mock_adapter):
            result = await _call(mcp_server, "tidus_complete_task", {
                "team_id": "test-team",
                "complexity": "simple",
                "domain": "chat",
                "estimated_input_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            })

        for field in ("content", "model_id", "vendor", "input_tokens",
                      "output_tokens", "cost_usd", "latency_ms"):
            assert field in result, f"Missing field: {field}"
