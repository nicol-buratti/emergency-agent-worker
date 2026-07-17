import json
import os
import logging
import operator
from typing import Literal, Optional, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph_swarm import (
    SwarmState,
    add_active_agent_router,
    create_handoff_tool,
)
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
app = None


class ThreatAssessment(BaseModel):
    room: str = Field(
        description="The room identifier for which the assessment is made."
    )
    danger: Literal["high", "medium", "low", "none"] = Field(
        description="Final assessed danger level based on all gathered data."
    )
    danger_type: Literal["fire", "smoke", "heat", "other", "none"] = Field(
        description="Type of danger identified, if any."
    )
    danger_score: float = Field(
        description="A numerical score representing the severity of the danger, on a scale from 0 to 1."
    )
    justification: str = Field(
        description="Brief justification for the assigned danger level."
    )


# class AgentState(TypedDict):
class AgentState(SwarmState):
    messages: Annotated[list, operator.add]
    recent_history: Optional[list[dict]]
    room_metadata: dict
    assessment: Optional[ThreatAssessment]


model = ChatOpenAI(
    model="tencent/hy3:free",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.2,
    max_retries=2,
    extra_body={
        "models": [
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "openai/gpt-oss-20b:free",
        ]
    },
)
fire_handoff = create_handoff_tool(
    agent_name="Fire Agent",
    description="Transfer to Fire Agent the hazard assessment",
)
earthquake_handoff = create_handoff_tool(
    agent_name="Earthquake Agent",
    description="Transfer to Earthquake Agent the hazard assessment",
)
formatter_handoff = create_handoff_tool(
    agent_name="Formatter Agent",
    description="Transfer to Formatter Agent the hazard assessment",
)
structured_model = model.with_structured_output(ThreatAssessment)
coordinator_model = model.bind_tools(
    [formatter_handoff, fire_handoff, earthquake_handoff]
)
fire_model = model.bind_tools([formatter_handoff])


async def coordinator_node(state: AgentState) -> dict:
    logger.info("[Coordinator] Initializing assessment.")
    room_metadata = state.get("room_metadata", {})
    messages = state["messages"]

    system_prompt = (
        "You are the Hazard Assessment Coordinator. Your task is to triage IoT data for building safety.\n\n"
        f"CURRENT ROOM METADATA:\n{json.dumps(room_metadata, indent=2)}\n\n"
        "OPERATIONAL INSTRUCTIONS:\n"
        "1. Identify the primary threat signature (e.g., thermal/smoke vs. seismic/vibration).\n"
        "2. Transfer the data and control to the appropriate expert agent (Fire Agent or Earthquake Agent).\n"
        "If no threat is detected, route the workflow directly to the Formatter with a 'none' danger assessment."
    )
    response = await coordinator_model.ainvoke(
        [SystemMessage(content=system_prompt)] + messages
    )

    return {"messages": [response]}


async def fire_agent_node(state: AgentState) -> dict:
    room_meta = state.get("room_metadata", {})
    logger.info(
        f"[Fire Agent] Analyzing thermal/smoke telemetry. Room: {room_meta.get('name')}"
    )
    messages = state["messages"]

    system_prompt = (
        "You are the Fire Safety Expert. Analyze the provided IoT telemetry specifically for "
        "fire-related hazards (e.g., temperature spikes, smoke presence, rapid heat rise). "
        "Formulate a clear final assessment detailing the danger level, specific danger type, "
        f"a severity score (0.0 to 1.0), and a concise justification. Room context: {room_meta}."
    )
    response = await model.ainvoke([SystemMessage(content=system_prompt)] + messages)

    logger.info(
        f"[Fire Agent] Analysis complete. Proceeding to Formatter. Room: {room_meta.get('name')}"
    )
    return {"messages": [response]}


async def earthquake_agent_node(state: AgentState) -> dict:
    room_meta = state.get("room_metadata", {})
    logger.info(
        f"[Earthquake Agent] Analyzing seismic/vibration telemetry. Room: {room_meta.get('name')}"
    )
    messages = state["messages"]

    system_prompt = (
        "You are the Earthquake Safety Expert. Analyze the provided IoT telemetry specifically for "
        "seismic hazards (e.g., abnormal vibrations, structural shifts, accelerometer anomalies). "
        "Formulate a clear final assessment detailing the danger level, specific danger type, "
        f"a severity score (0.0 to 1.0), and a concise justification. Room context: {room_meta}."
    )

    response = await model.ainvoke([SystemMessage(content=system_prompt)] + messages)

    logger.info(
        f"[Earthquake Agent] Analysis complete. Proceeding to Formatter. Room: {room_meta.get('name')}"
    )
    return {"messages": [response]}


async def formatter_node(state: AgentState) -> ThreatAssessment:
    logger.info("[Formatter] Extracting structured ThreatAssessment.")
    messages = state["messages"]

    assessment_result = await structured_model.ainvoke(
        f"Extract the final assessment from this conversation: {messages[-1].content}"
    )
    assessment_result.room = state.get("room_metadata", {}).get("name", "unknown")

    logger.info(f"[Formatter] Assessment finalized: {assessment_result.model_dump()}")
    return {"assessment": assessment_result}


async def build_graph():
    # coordinator_agent = (
    #     StateGraph(AgentState)
    #     .add_node("Coordinator", coordinator_node)
    #     .add_edge(START, "Coordinator")
    #     .add_edge("Coordinator", END)
    # ).compile(name="Coordinator")
    # fire_agent = (
    #     StateGraph(AgentState)
    #     .add_node("Fire Agent", fire_agent_node)
    #     .add_edge(START, "Fire Agent")
    #     .add_edge("Fire Agent", END)
    # ).compile()
    # earthquake_agent = (
    #     StateGraph(AgentState)
    #     .add_node("Earthquake Agent", earthquake_agent_node)
    #     .add_edge(START, "Earthquake Agent")
    #     .add_edge("Earthquake Agent", END)
    # ).compile()
    # formatter_agent = (
    #     StateGraph(AgentState)
    #     .add_node("Formatter", formatter_node)
    #     .add_edge(START, "Formatter")
    #     .add_edge("Formatter", END)
    # ).compile(name="Formatter")

    workflow = (
        StateGraph(AgentState)
        .add_node(
            "Coordinator",
            coordinator_node,
            destinations=(
                "Formatter",
                "Earthquake Agent",
                "Fire Agent",
                # get_handoff_destinations(formatter_agent),
                # get_handoff_destinations(fire_agent),
                # get_handoff_destinations(earthquake_agent),
            ),
        )
        .add_node(
            "Fire Agent",
            fire_agent_node,
            destinations=("Formatter",),
        )
        .add_node(
            "Earthquake Agent",
            earthquake_agent_node,
            destinations=("Formatter",),
        )
        .add_node(
            "Formatter",
            formatter_node,
            destinations=(END,),
        )
    )
    workflow = add_active_agent_router(
        builder=workflow,
        route_to=["Coordinator", "Formatter", "Earthquake Agent", "Fire Agent"],
        default_active_agent="Coordinator",
    )

    global app
    app = workflow.compile()
    logger.info("\n%s", app.get_graph().draw_ascii())
    _ = app.get_graph().draw_mermaid_png()


async def call_agent(data):
    logger.info("[Agent] Starting LangGraph call.")

    prompt = (
        f"Initial IoT Data:\n{data}\n\n"
        "Instructions:\n"
        "1. The 'room' field in the data indicates the device name.\n"
        "2. Evaluate the readings for anomalies.\n"
        "Analysis Guidelines:\n"
        "- Timestamps are in Unix Epoch format.\n"
        "- TVOC values of 60,000 indicate sensor saturation.\n"
    )
    initial_state: AgentState = {
        "messages": [HumanMessage(content=prompt)],
        "recent_history": data.get("sensor_data", []),
        "room_metadata": {
            "name": data.get("room", "unknown"),
        },
        "assessment": None,
    }

    config = {"configurable": {"thread_id": "1"}}
    result = await app.ainvoke(initial_state, config=config)
    # result = result["assessment"]
    logger.info(result)
    logger.info("[Agent] Execution complete.")
    return result["assessment"]
