import logging
import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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
            ],
            system_prompt="""You are the Hazard Assessment Coordinator.
Triage the provided IoT data or use available tools to assess room safety.
Transfer control to 'Fire Agent' or 'Earthquake Agent' if anomalies exist.
If data shows normal conditions, terminate directly with a safe assessment.""",
            name="Coordinator",
        )

        fire_agent = create_agent(
            self.model,
            tools=[*tools, end_handoff],
            system_prompt="""You are the Fire Safety Expert. Analyze telemetry for fire hazards (heat, smoke, CO2 spikes).
Formulate a clear diagnosis and terminate the swarm once ready.""",
            name="Fire Agent",
        )

        earthquake_agent = create_agent(
            self.model,
            tools=[*tools, end_handoff],
            system_prompt="""You are the Earthquake Safety Expert. Analyze telemetry for seismic activity (vibrations, acceleration).
Formulate a clear diagnosis and terminate the swarm once ready.""",
            name="Earthquake Agent",
        )

        workflow = create_swarm(
            [coordinator, fire_agent, earthquake_agent],
            default_active_agent="Coordinator",
        )

        checkpointer = MemorySaver()
        self.app = workflow.compile(checkpointer=checkpointer, debug=debug)
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

        logger.info(
            "[Formatter] -> Formatting final assessment into Pydantic schema..."
        )
        extraction_prompt = [
            *result["messages"],
            HumanMessage(
                content=f"Summarize the conversation and output the final ThreatAssessment structured JSON for room '{thread_id}'."
            ),
        ]

        structured_res: ThreatAssessment = await self.structured_model.ainvoke(
            extraction_prompt
        )

        # Cleaning checkpoint of the thread
        await self.app.checkpointer.adelete_thread(thread_id=thread_id)

        return structured_res.model_dump()
