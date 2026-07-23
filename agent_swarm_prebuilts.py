import logging
import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph_swarm import SwarmState, create_handoff_tool, create_swarm
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

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
    warning: Literal["none", "pre-alert"] = Field(
        description="Early warning indicator for suspicious data that may precede a danger"
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
        self.formatter_system_prompt = SystemMessage(
            content="""You are the Formatter Agent for the Hazard Assessment Swarm.
        Your sole responsibility is to review the conversation history, diagnostics, and telemetry data gathered by the other Agents.
        Synthesize their findings and formulate the final safety assessment. Accurately translate the experts' consensus—including identified hazard types, danger levels, and early warnings—into the required structured output.
        Provide a concise, evidence-based justification for the final verdict based strictly on the preceding investigation, without conducting any new analysis of your own.""",
        )

        self.app = None

    async def initialize_graph(
        self, tools: list = None, debug: bool = False, print_agent=False
    ):
        """Initialize and build the agent swarm"""
        tools = tools or []

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
            ],
            system_prompt="""You are the Hazard Assessment Coordinator.
Triage the provided IoT data or use available tools to assess room safety.
Transfer control to 'Fire Agent' or 'Earthquake Agent' if anomalies exist.
If data shows normal conditions, terminate directly with a safe assessment.""",
            name="Coordinator",
        )

        fire_agent = create_agent(
            self.model,
            tools=[*tools],
            system_prompt="""You are the Fire Safety Expert. Analyze telemetry for fire hazards (heat, smoke, CO2 spikes).
Formulate a clear diagnosis and terminate the swarm once ready.""",
            name="Fire Agent",
        )

        earthquake_agent = create_agent(
            self.model,
            tools=[*tools],
            system_prompt="""You are the Earthquake Safety Expert. Analyze telemetry for seismic activity (vibrations, acceleration).
Formulate a clear diagnosis and terminate the swarm once ready.""",
            name="Earthquake Agent",
        )

        workflow = create_swarm(
            [coordinator, fire_agent, earthquake_agent],
            default_active_agent="Coordinator",
        )

        checkpointer = InMemorySaver()
        self.app = workflow.compile(checkpointer=checkpointer, debug=debug)
        if print_agent:
            logger.info(self.app.get_graph().draw_ascii())

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
        # Formatting prediction
        assessment: ThreatAssessment = await self.structured_model.ainvoke(
            [self.formatter_system_prompt, result["messages"][-1]]
        )

        return assessment.model_dump()
