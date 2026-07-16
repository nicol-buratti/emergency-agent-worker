import os
import logging
from typing import Literal, TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

llm = None
app = None
tools = None  # Added global to hold fetched tools
tool_node = None  # Added global to hold the ToolNode instance
mcp_client = None  # The client will now maintain the SSE connection as long as the application runs


class ThreatAssessment(BaseModel):
    room: str = Field(
        description="The room identifier for which the assessment is made."
    )
    danger: Literal["high", "medium", "low", "unsure"] = Field(
        description="Final assessed danger level based on all gathered data."
    )
    danger_type: Literal["fire", "smoke", "heat", "other"] = Field(
        description="Type of danger identified, if any."
    )
    danger_score: float = Field(
        description="A numerical score representing the severity of the danger, on a scale from 0 to 1."
    )
    justification: str = Field(
        description="Brief justification for the assigned danger level."
    )


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    assessment: dict


async def reason(state: AgentState) -> dict:
    logger.info("   [Agent] -> Reasoning with LangGraph...")
    llm_with_tools = llm.bind_tools(tools)

    # Prepend instructions to the message history dynamically
    system_msg = SystemMessage(content="Analyze the IoT data for fire risk. ,.")
    messages = [system_msg] + state["messages"]

    result = await llm_with_tools.ainvoke(messages)
    return {"messages": [result]}


def route_reasoning(
    state: AgentState,
) -> Literal["fetch_external_data", "extract_final_assessment"]:
    last_message = state["messages"][-1]

    # Route to tools if the LLM generated a tool call
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "fetch_external_data"

    # Route to final extraction if the LLM is finished
    return "extract_final_assessment"


def safe_tool_node(state: AgentState):
    logger.info("   [Agent] -> Invoking tools with LangGraph...")
    try:
        # tool_node is the instantiated ToolNode(tools)
        return tool_node.invoke(state)
    except Exception as e:
        last_message = state["messages"][-1]
        error_messages = []

        # Acknowledge the failed tool calls so LangGraph can proceed
        for tool_call in last_message.tool_calls:
            error_messages.append(
                ToolMessage(
                    content=f"Tool execution failed with error: {str(e)}. Review your arguments and try again.",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        return {"messages": error_messages}


async def extract_final_assessment(state: AgentState) -> dict:
    logger.info("   [Agent] -> Extracting final assessment with LangGraph...")
    structured_llm = llm.with_structured_output(ThreatAssessment)

    system_msg = SystemMessage(
        content="Based on the entire conversation history, extract the final fire threat assessment."
    )
    messages = [system_msg] + state["messages"]

    result = await structured_llm.ainvoke(messages)
    return {"assessment": result.model_dump()}


async def build_graph():
    global mcp_client, llm, tools, app, tool_node
    mcp_client = MultiServerMCPClient(
        {
            "thingsboard": {"url": "http://localhost:8000/sse", "transport": "sse"},
            "ddgs": {"command": "ddgs", "args": ["mcp"], "transport": "stdio"},
        }
    )

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.2,
        max_retries=2,
    )

    tools = await mcp_client.get_tools()
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("reason", reason)
    workflow.add_node("fetch_external_data", safe_tool_node)
    workflow.add_node("extract_final_assessment", extract_final_assessment)

    workflow.add_edge(START, "reason")
    workflow.add_conditional_edges("reason", route_reasoning)
    workflow.add_edge("fetch_external_data", "reason")
    workflow.add_edge("extract_final_assessment", END)

    app = workflow.compile()
    logger.info("\n%s", app.get_graph().draw_ascii())


async def call_agent(data):
    logger.info("   [Agent] -> Starting LangGraph call...")

    prompt = f"Given this IoT data, evaluate if there is a risk of fires:\n\n{data}"
    initial_state = {"messages": [HumanMessage(content=prompt)]}

    result = await app.ainvoke(initial_state)
    logger.info("   [Agent] -> Final Assessment: %s", result.get("assessment"))

    return result.get("assessment")
