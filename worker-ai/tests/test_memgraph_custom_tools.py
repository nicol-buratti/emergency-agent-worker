"""Unit tests for the Memgraph tool wrappers.

These tests mock the MCP client's `run_query` tool so they run without a
live Memgraph/MCP server. They mainly guard the Cypher string-building
logic (parameter interpolation, LIMIT/depth handling) since that's the
part most likely to silently break — e.g. an off-by-one in the hop range,
or a room name that breaks the query.

A separate, slower integration suite (marked `@pytest.mark.integration`)
should run these same tools against a real Memgraph loaded with a small
fixture graph (see docker-compose.yml) to catch Cypher errors that only
show up against a real database.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.memgraph_custom_tools import get_memgraph_tools


@pytest.fixture
async def tools_and_query_mock():
    """Builds the tool set with the underlying MCP `run_query` tool
    replaced by an AsyncMock, and returns (tools_by_name, query_mock).
    """
    fake_query_tool = AsyncMock()
    fake_query_tool.name = "run_query"
    fake_query_tool.ainvoke.return_value = "[]"

    fake_mcp_client = AsyncMock()
    fake_mcp_client.get_tools.return_value = [fake_query_tool]

    with patch(
        "src.memgraph_custom_tools.MultiServerMCPClient", return_value=fake_mcp_client
    ):
        tools = await get_memgraph_tools()

    return {t.name: t for t in tools}, fake_query_tool


class TestToolSetShape:
    @pytest.mark.asyncio
    async def test_returns_all_six_expected_tools(self, tools_and_query_mock):
        tools_by_name, _ = tools_and_query_mock

        assert set(tools_by_name) == {
            "get_room_data",
            "get_adjacent_rooms",
            "get_hvac_and_shaft_connections",
            "get_vertical_topology",
            "get_nearby_critical_hazards",
            "get_structural_context",
        }

    @pytest.mark.asyncio
    async def test_raises_if_run_query_tool_is_missing(self):
        """get_memgraph_tools() does `next(t for t in mcp_tools if t.name ==
        'run_query')`; if the MCP server doesn't expose that tool, this
        should fail fast at startup rather than later with a confusing
        AttributeError deep inside a query call."""
        fake_mcp_client = AsyncMock()
        fake_mcp_client.get_tools.return_value = []  # no run_query tool

        with patch(
            "src.memgraph_custom_tools.MultiServerMCPClient",
            return_value=fake_mcp_client,
        ):
            # Note: PEP 479 turns a StopIteration raised inside a coroutine
            # into a RuntimeError, so that's what actually propagates here.
            with pytest.raises(RuntimeError):
                await get_memgraph_tools()


class TestGetRoomData:
    @pytest.mark.asyncio
    async def test_builds_query_with_department_and_room_filters(
        self, tools_and_query_mock
    ):
        tools_by_name, query_mock = tools_and_query_mock

        await tools_by_name["get_room_data"].ainvoke(
            {"department": "ICU", "room": "room-101"}
        )

        cypher = query_mock.ainvoke.await_args.args[0]["query"]
        assert "n.department = 'ICU'" in cypher
        assert "n.name = 'room-101'" in cypher
        assert "LIMIT 1" in cypher


class TestGetAdjacentRooms:
    @pytest.mark.asyncio
    async def test_builds_query_with_requested_depth_and_limit(
        self, tools_and_query_mock
    ):
        tools_by_name, query_mock = tools_and_query_mock

        await tools_by_name["get_adjacent_rooms"].ainvoke(
            {"department": "ICU", "room": "room-101", "depth": 2, "limit": 10}
        )

        cypher = query_mock.ainvoke.await_args.args[0]["query"]
        assert "*1..2" in cypher
        assert "LIMIT 10" in cypher
        assert "n.department = 'ICU'" in cypher
        assert "n.name = 'room-101'" in cypher

    @pytest.mark.asyncio
    async def test_depth_one_means_direct_neighbors_only(self, tools_and_query_mock):
        tools_by_name, query_mock = tools_and_query_mock

        await tools_by_name["get_adjacent_rooms"].ainvoke(
            {"department": "ICU", "room": "room-101", "depth": 1, "limit": 5}
        )

        cypher = query_mock.ainvoke.await_args.args[0]["query"]
        assert "*1..1" in cypher


class TestPlaceholderTools:
    """get_hvac_and_shaft_connections, get_vertical_topology,
    get_nearby_critical_hazards and get_structural_context are currently
    stubs that don't query Memgraph at all. These tests pin down that
    documented (if temporary) behavior so a future implementation change
    is a deliberate test update, not a silent regression."""

    @pytest.mark.asyncio
    async def test_hvac_tool_returns_placeholder_and_does_not_query(
        self, tools_and_query_mock
    ):
        tools_by_name, query_mock = tools_and_query_mock

        result = await tools_by_name["get_hvac_and_shaft_connections"].ainvoke(
            {"room": "room-101"}
        )

        assert "No real data" in result
        query_mock.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_vertical_topology_tool_returns_placeholder(
        self, tools_and_query_mock
    ):
        tools_by_name, _ = tools_and_query_mock

        result = await tools_by_name["get_vertical_topology"].ainvoke(
            {"room": "room-101"}
        )

        assert "No real data" in result

    @pytest.mark.asyncio
    async def test_nearby_critical_hazards_tool_returns_placeholder(
        self, tools_and_query_mock
    ):
        tools_by_name, _ = tools_and_query_mock

        result = await tools_by_name["get_nearby_critical_hazards"].ainvoke(
            {"room": "room-101", "radius_hops": 2}
        )

        assert "No real data" in result

    @pytest.mark.asyncio
    async def test_structural_context_tool_returns_placeholder(
        self, tools_and_query_mock
    ):
        tools_by_name, _ = tools_and_query_mock

        result = await tools_by_name["get_structural_context"].ainvoke(
            {"room": "room-101"}
        )

        assert "No real data" in result
