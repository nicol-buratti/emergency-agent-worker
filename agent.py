import os
import logging
from typing import Literal, TypedDict, Annotated
import operator
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Given this IoT data, evaluate if there is a risk of fires. 
If the data is missing critical context (like weather, historical patterns, or specific sensor baselines), 
mark yourself as unsure and request external data.
"""
llm = None  # Global variable to hold the LLM instance
app = None  # Global variable to hold the compiled graph app


# 1. Update structured output schema to allow uncertainty
class ThreatAssessment(BaseModel):
    danger: Literal["high", "medium", "low", "unsure"] = Field(
        description="Assessed danger level. Use 'unsure' if you lack sufficient data."
    )
    needs_external_data: bool = Field(
        description="Set to True if you need to query an external database, MCP server, or the internet."
    )
    search_query: str | None = Field(
        default=None,
        description="The query to search for if needs_external_data is True (e.g., 'current humidity in zone 4').",
    )


# 2. Update state to use Annotated for message appending
class AgentState(TypedDict):
    # Using Annotated and operator.add ensures messages are appended, not overwritten
    messages: Annotated[list, operator.add]
    assessment: dict


# 3. Define the LLM node logic
async def assess_danger(state: AgentState) -> dict:
    structured_llm = llm.with_structured_output(ThreatAssessment)

    # We pass the entire conversation history to the LLM so it has the context of previous searches
    prompt = (
        f"Analyze the data and determine the fire danger level:\n\n{state['messages']}"
    )

    result = await structured_llm.ainvoke(prompt)

    return {"assessment": result.model_dump()}


# 5. Define the routing logic
def route_assessment(state: AgentState) -> Literal["fetch_external_data", "__end__"]:
    assessment = state["assessment"]

    # If the LLM explicitly asked for data or marked danger as unsure, route to the tool
    if assessment.get("needs_external_data") or assessment.get("danger") == "unsure":
        return "fetch_external_data"

    # Otherwise, we have a confident assessment, so we end.
    return "__end__"


async def build_graph():
    mcp_client = MultiServerMCPClient(
        {
            "thingsboard": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            "ddgs": {
                "command": "ddgs",
                "args": ["mcp"],
                "transport": "stdio",
            },
        }
    )

    global llm
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.2,  # Lower temperature is usually better for strict classification (to be verified)
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    tools = await mcp_client.get_tools()

    # 6. Construct graph routing
    workflow = StateGraph(AgentState)

    tool_node = ToolNode(tools)

    workflow.add_node("assess", assess_danger)
    # workflow.add_node("fetch_external_data", fetch_external_data)
    workflow.add_node("fetch_external_data", tool_node)

    workflow.add_edge(START, "assess")

    # Add the conditional routing
    workflow.add_conditional_edges("assess", route_assessment)

    # After fetching data, loop back to assess again with the new context
    workflow.add_edge("fetch_external_data", "assess")

    global app
    app = workflow.compile()

    # Print the graph in ASCII format in the terminal
    ascii_art = app.get_graph().draw_ascii()
    logger.info("\n%s", ascii_art)


# 7. Execution function
async def call_agent(data):
    logger.info("   [Agent] -> Starting LangGraph call...")

    complete_prompt = f"{PROMPT_TEMPLATE}\n\n{data}"
    initial_state = {"messages": [HumanMessage(content=complete_prompt)]}

    try:
        result = await app.ainvoke(initial_state)
        logger.info("   [Agent] -> Final response received from LLM!")
        logger.info("   [Agent] -> Assessment: %s", result["assessment"])

        return result["assessment"]

    except Exception as e:
        logger.error("   [Agent] -> FATAL ERROR inside LangGraph: %s", e)
        raise
