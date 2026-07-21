from langchain_openai import ChatOpenAI
import os
from langchain.agents import create_agent
from langgraph_swarm import SwarmState, create_handoff_tool, create_swarm
from langchain_core.messages import HumanMessage
import logging
from typing import Literal, Optional, Annotated
import operator
from langgraph.graph import END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
app = None
model = ChatOpenAI(
    # model=os.getenv("LLM_MODEL"),
    model="tencent/hy3:free",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.2,
    max_retries=2,
    extra_body={
        "models": [
            # "tencent/hy3:free",
            "google/gemma-4-26b-a4b-it:free",
            # "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "openai/gpt-oss-20b:free",
        ]
    },
)


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


class AgentState(SwarmState):
    messages: Annotated[list, operator.add]
    recent_history: Optional[list[dict]]
    room_metadata: dict
    assessment: Optional[ThreatAssessment]


agent_model = model.with_structured_output(ThreatAssessment)


async def build_graph() -> ThreatAssessment:
    # mcp_client_thingsboard = MultiServerMCPClient(
    #     {
    #         "thingsboard": {"url": "http://localhost:8000/sse", "transport": "sse"},
    #         # "ddgs": {"command": "ddgs", "args": ["mcp"], "transport": "stdio"},
    #     }
    # )

    # thingsboard_tools = await mcp_client_thingsboard.get_tools()

    coordinator = create_agent(
        model,
        tools=[
            # *thingsboard_tools,
            create_handoff_tool(
                agent_name="Fire Agent",
                description="Transfer to Fire Agent the hazard assessment",
            ),
            create_handoff_tool(
                agent_name="Earthquake Agent",
                description="Transfer to Earthquake Agent the hazard assessment",
            ),
            create_handoff_tool(
                agent_name=END,
                description="Terminate the swarm and output the final assessment",
            ),
        ],
        system_prompt="""You are the Hazard Assessment Coordinator. Your task is to triage IoT data for building safety.

1) Use your tools to fetch telemetry for the given device/room over the last 5 minutes.
2) Identify the primary threat signature (e.g., thermal/smoke vs. seismic/vibration).
3) Transfer the data and control to the appropriate expert agent (Fire Agent or Earthquake Agent).

If no threat is detected, terminate the swarm directly with a 'none' danger assessment.""",
        name="Coordinator",
        # debug=True,
    )

    fire_agent = create_agent(
        model,
        tools=[
            # *thingsboard_tools,
        ],
        system_prompt="""You are the Fire Safety Expert. Analyze the provided IoT telemetry specifically for fire-related hazards (e.g., temperature spikes, smoke presence, rapid heat rise).

Formulate a clear final assessment detailing:
- Danger level
- Specific danger type
- Severity score (0.0 to 1.0)
- Concise justification

Once complete, terminate the swarm to output the final assessment.""",
        name="Fire Agent",
        # debug=True,
    )

    earthquake_agent = create_agent(
        model,
        tools=[
            # *thingsboard_tools,
        ],
        system_prompt="""You are the Earthquake Safety Expert. Analyze the provided IoT telemetry specifically for seismic hazards (e.g., abnormal vibrations, structural shifts, accelerometer anomalies).

Formulate a clear final assessment detailing:
- Danger level
- Specific danger type
- Severity score (0.0 to 1.0)
- Concise justification

Once complete, terminate the swarm to output the final assessment.""",
        name="Earthquake Agent",
        # debug=True,
    )

    workflow = create_swarm(
        [coordinator, fire_agent, earthquake_agent],
        default_active_agent="Coordinator",
    )
    checkpointer = MemorySaver()
    global app
    app = workflow.compile(checkpointer=checkpointer, debug=True)
    logger.info("\n%s", app.get_graph().draw_ascii())
    _ = app.get_graph().draw_mermaid_png()
    # with open("grafo_langgraph.png", "wb") as f:
    #     f.write(png_bytes)


async def call_agent(data):
    logger.info("[Agent] -> Starting LangGraph call...")

    # data = """[
    #   {
    #     "timestamp": "2026-07-17T22:45:00Z",
    #     "temperature_celsius": 22.4,
    #     "co2_ppm": 415
    #   },
    #   {
    #     "timestamp": "2026-07-17T22:46:00Z",
    #     "temperature_celsius": 28.1,
    #     "co2_ppm": 950
    #   },
    #   {
    #     "timestamp": "2026-07-17T22:47:00Z",
    #     "temperature_celsius": 46.8,
    #     "co2_ppm": 2800
    #   },
    #   {
    #     "timestamp": "2026-07-17T22:48:00Z",
    #     "temperature_celsius": 78.5,
    #     "co2_ppm": 5500
    #   },
    #   {
    #     "timestamp": "2026-07-17T22:49:00Z",
    #     "temperature_celsius": 124.0,
    #     "co2_ppm": 9200
    #   }
    # ]"""
    thread_id = data["room"]
    config = {"configurable": {"thread_id": thread_id}}

    # 1. Verifica se esiste un checkpoint salvato per questo thread
    current_state = await app.aget_state(config)

    if current_state.values:
        logger.info(
            f"Previous state detected for thread_id={thread_id}. Resuming execution..."
        )
        result = await app.ainvoke(None, config=config)
    else:
        logger.info("No status found. Starting a new execution...")

        prompt = (
            f"Initial IoT Data:\n{data}\n\n"
            "Instructions:\n"
            "1. The 'room' field in the data indicates the device name.\n"
            "2. Use your tools to query this device's telemetry for the last 5 minutes.\n"
            "3. Evaluate the readings for anomalies.\n"
            "4. Route to the appropriate expert agent if a threat is suspected, or terminate with a safe baseline assessment.\n"
            "Analysis Guidelines:\n"
            "- Timestamps are in Unix Epoch format.\n"
            "- TVOC values of 60,000 indicate sensor saturation.\n"
        )

        initial_state: AgentState = {
            "messages": [HumanMessage(content=prompt)],
            "recent_history": [],
            "room_metadata": {},
            "assessment": None,
        }

        result = await app.ainvoke(initial_state, config=config)

    # result = list(initial_state["messages"])
    # async for event in app.astream(initial_state, config=config, stream_mode="updates"):
    #     for node_name, node_state in event.items():
    #         logger.info(f"[{node_name}] -> Agent in execution...")
    #         if "messages" in node_state:
    #             result.extend(node_state["messages"])

    logger.info("[Formatter] -> Formatting final assessment into Pydantic schema...")
    result = await agent_model.ainvoke(
        result["messages"]
        + [
            HumanMessage(
                content="Extract the structural threat assessment from this conversation summary"
            )
        ]
    )
    await app.checkpointer.adelete_thread(thread_id=thread_id)
    logger.info("[Agent] -> Swarm completed.")
    return result.model_dump()
