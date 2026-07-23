from langchain_mcp_adapters.client import MultiServerMCPClient

from typing import List
from langchain_core.tools import tool, BaseTool


async def get_memgraph_tools() -> List[BaseTool]:
    """Connects to the Memgraph MCP server and returns a list of high-level LangChain/LangGraph tools."""
    mcp_client = MultiServerMCPClient(
        {
            "memgraph": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            },
        }
    )
    mcp_tools = await mcp_client.get_tools()
    query_tool = next(t for t in mcp_tools if t.name == "run_query")

    @tool
    async def get_room_data(room: str) -> str:
        """Finds a specific room in Memgraph and returns its properties."""
        cypher = f"""
        MATCH (n:Place)
        WHERE n.name CONTAINS '{room}'
        RETURN n
        LIMIT 1
        """
        return await query_tool.ainvoke({"query": cypher})

    @tool
    async def get_adjacent_rooms(room: str, depth: int, limit: int) -> str:
        """Explores connected rooms in the graph starting from a room up to a given depth integer."""
        cypher = f"""
        MATCH (n:Place)
        WHERE n.name CONTAINS '{room}'

        MATCH p = (n)-[:CONNECTED_TO*1..{depth}]-(target)

        UNWIND range(0, length(p) - 1) AS i
        WITH
        relationships(p)[i] AS rel,
        nodes(p)[i] AS n_from,
        nodes(p)[i+1] AS n_to,
        i AS level

        ORDER BY level ASC

        WITH rel, collect({{n1: n_from, n2: n_to, depth: level}})[0] AS data

        RETURN
        data.n1.name AS n1,
        type(rel) AS edge,
        data.n2.name AS n2
        ORDER BY data.depth ASC, n1 ASC, n2 ASC
        LIMIT {limit}
        """
        return await query_tool.ainvoke({"query": cypher})

    return [get_room_data, get_adjacent_rooms]
