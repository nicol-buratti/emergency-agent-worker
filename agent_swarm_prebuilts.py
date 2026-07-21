import logging
import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph_swarm import SwarmState, create_handoff_tool, create_swarm
from pydantic import BaseModel, Field
from langchain.agents import create_agent

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic schemas and State
# ------------------------------------------------------------------
class ThreatAssessment(BaseModel):
    room: str = Field(
        description="The room identifier for which the assessment is made."
    )
    danger: Literal["high", "medium", "low", "none"] = Field(
        description="Final assessed danger level based on all gathered data."
    )
    danger_type: Literal["fire", "smoke", "heat", "earthquake", "other", "none"] = (
        Field(description="Type of danger identified, if any.")
    )
    danger_score: float = Field(
        description="Severity score on a scale from 0.0 to 1.0."
    )
    justification: str = Field(
        description="Brief justification for the assigned danger level."
    )


class AgentState(SwarmState):
    messages: Annotated[list, operator.add]


# ------------------------------------------------------------------
# Swarm manager Class
# ------------------------------------------------------------------
class HazardSwarmManager:
    def __init__(self):
        self.model = ChatOpenAI(
            model="google/gemma-4-26b-a4b-it:free",
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0.2,
            max_retries=2,
            extra_body={
                "models": [
                    "tencent/hy3:free",
                    "nvidia/nemotron-nano-9b-v2:free",
                    "openai/gpt-oss-20b:free",
                ]
            },
        )
        self.structured_model = self.model.with_structured_output(ThreatAssessment)
        self.app = None

    async def initialize_graph(self, tools: list = None, debug: bool = False):
        """Initialize and build the agent swarm"""
        tools = tools or []

        # Shared Termination Tool for Specialists
        end_handoff = create_handoff_tool(
            agent_name=END,
            description="Terminate the swarm and finalize assessment.",
        )

        coordinator = create_agent(
            self.model,
            tools=[
                *tools,
                create_handoff_tool(
                    agent_name="Fire Agent",
                    description="Transfer to Fire Agent for thermal/smoke hazard analysis.",
                ),
                create_handoff_tool(
                    agent_name="Earthquake Agent",
                    description="Transfer to Earthquake Agent for seismic/vibration analysis.",
                ),
                end_handoff,
                ThreatAssessment,
            ],
            system_prompt=f"""You are the Hazard Assessment Coordinator.
Triage the provided IoT data or use available tools to assess room safety.
Transfer control to 'Fire Agent' or 'Earthquake Agent' if anomalies exist.
If data shows normal conditions, terminate directly with a safe assessment.
If no specialized agent is needed, the response output must follow the {ThreatAssessment.__name__} structure.""",
            name="Coordinator",
        )

        fire_agent = create_agent(
            self.model,
            tools=[*tools, end_handoff, ThreatAssessment],
            system_prompt=f"""You are the Fire Safety Expert. Analyze telemetry for fire hazards (heat, smoke, CO2 spikes).
Formulate a clear diagnosis and terminate the swarm once ready.
The response output must follow the {ThreatAssessment.__name__} structure.""",
            name="Fire Agent",
        )

        earthquake_agent = create_agent(
            self.model,
            tools=[*tools, end_handoff, ThreatAssessment],
            system_prompt=f"""You are the Earthquake Safety Expert. Analyze telemetry for seismic activity (vibrations, acceleration).
Formulate a clear diagnosis and terminate the swarm once ready.
The response output must follow the {ThreatAssessment.__name__} structure.""",
            name="Earthquake Agent",
        )

        workflow = create_swarm(
            [coordinator, fire_agent, earthquake_agent],
            default_active_agent="Coordinator",
        )

        self.app = workflow.compile(debug=debug)
        logger.info("Swarm graph compiled successfully.")

    async def process_data(self, data: dict) -> dict:
        if not self.app:
            await self.initialize_graph()

        thread_id = str(data.get("room", "default_room"))
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(f"[Agent] -> Executing Swarm for room: {thread_id}")

        prompt = (
            f"IoT Context & Input Data:\n{data}\n\n"
            "Instructions:\n"
            "1. Evaluate the reading values for safety/threat levels.\n"
            "2. Hand over to specialized agents if needed.\n"
            "3. Conclude the assessment."
        )

        initial_state: AgentState = {"messages": [HumanMessage(content=prompt)]}

        # Swarm execution
        result = await self.app.ainvoke(initial_state, config=config)

        # Search for the latest AI message (AIMessage) that contains a call to the ThreatAssessment tool
        assessment_data = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == ThreatAssessment.__name__:
                        assessment_data = tool_call["args"]
                        break
            if assessment_data:
                break

        if not assessment_data:
            # Safety fallback in case the LLM did not use the tool
            logger.warning("The ThreatAssessment tool was not called.")
            return {}

        # Native Pydantic Validation
        assessment = ThreatAssessment(**assessment_data)

        return assessment.model_dump()
