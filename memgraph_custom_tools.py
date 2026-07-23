from langchain_mcp_adapters.client import MultiServerMCPClient

from typing import List
from langchain_core.tools import tool, BaseTool


async def get_memgraph_tools() -> List[BaseTool]:
    """Connects to the Memgraph MCP server and returns a list of high-level LangChain/LangGraph tools."""
    # 1. Connect to the underlying Memgraph MCP server
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

    # 2. Define custom tools using the @tool decorator
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

        MATCH p = (n)-[*1..{depth}]-(destinazione)

        UNWIND range(0, length(p) - 1) AS i
        WITH
          relationships(p)[i] AS rel,
          nodes(p)[i] AS n_from,
          nodes(p)[i+1] AS n_to,
          i AS livello

        ORDER BY livello ASC

        WITH rel, collect({{n1: n_from, n2: n_to, strato: livello}})[0] AS dati

        RETURN
          dati.n1.name AS nodo1,
          type(rel) AS edge,
          dati.n2.name AS nodo2
        ORDER BY dati.strato ASC, nodo1 ASC, nodo2 ASC
        LIMIT {limit}
        """
        return await query_tool.ainvoke({"query": cypher})

    # 3. Return the tools as a list
    return [get_room_data, get_adjacent_rooms]
