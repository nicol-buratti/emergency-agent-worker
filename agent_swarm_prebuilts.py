import logging
import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph_swarm import SwarmState, create_handoff_tool, create_swarm
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic schemas and State
# ------------------------------------------------------------------
class ThreatAssessment(BaseModel):
    """Class that represent the threat assessment of a room"""

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
    """Manages the lifecycle and execution of the LangGraph-based hazard assessment swarm."""

    def __init__(self):
        self.model = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0.2,
            max_retries=2,
            extra_body={"models": os.getenv("EXTRA_LLM_MODELS")},
        )
        self.callbacks = []
        if os.getenv("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "t", "yes"):
            self.callbacks.append(CallbackHandler())
        template_string = """IoT Context & Input Data:
{data}

Instructions:
1. Evaluate the reading values for safety/threat levels.
2. Hand over to specialized agents if needed.
3. Conclude the assessment."""

        self.prompt_template = PromptTemplate(
            input_variables=["data"], template=template_string
        )

        self.app = None

    def _extract_assessment(result):
        assessment_data = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == ThreatAssessment.__name__:
                        assessment_data = tool_call["args"]
                        break
            if assessment_data:
                break
        return assessment_data

    async def initialize_graph(
        self, tools: list = None, debug: bool = False, print_agent=False
    ) -> None:
        """Initialize and build the agent swarm"""
        tools = tools or []

        coordinator_prompt = PromptTemplate(
            input_variables=[],
            template="""You are the Hazard Assessment Coordinator.
Triage the provided IoT data to assess room safety. When necessary, use the provided Memgraph database tools to query the building's topology and inspect node parameters to understand the spatial layout and current sensor snapshot.
Transfer control to the specialized Agents if anomalies exist.
If both the telemetry data and the building's node parameters indicate normal, safe conditions, terminate directly with a safe assessment.""",
        )

        fire_prompt = PromptTemplate(
            input_variables=["format_name"],
            template="""You are the Fire Safety Expert. Analyze telemetry for fire hazards (heat, smoke, CO2 spikes).
If anomalous data is detected, use the Memgraph database tools to check the building's topology and node parameters. You must identify adjacent rooms, ventilation paths, or connected structural nodes to assess the risk of fire and smoke spread.
Formulate a clear diagnosis based on both the telemetry and the building's spatial graph, and terminate the swarm once ready.
Output the final response by using {format_name} format.""",
        )

        earthquake_prompt = PromptTemplate(
            input_variables=["format_name"],
            template="""You are the Earthquake Safety Expert. Analyze telemetry for seismic activity (vibrations, acceleration, structural shifts).
When assessing seismic impact, use the Memgraph database tools to query the building's topology and structural node parameters. You must understand load-bearing dependencies, material parameters, and damage propagation across connected building elements.
Formulate a clear diagnosis based on the telemetry and the building's structural graph, and terminate the swarm once ready.
Output the final response by using {format_name} format.""",
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
            ],
            response_format=ThreatAssessment,
            system_prompt=coordinator_prompt.format(),
            name="Coordinator",
        )

        fire_agent = create_agent(
            self.model,
            tools=[*tools],
            response_format=ThreatAssessment,
            system_prompt=fire_prompt.format(format_name=ThreatAssessment.__name__),
            name="Fire Agent",
        )

        earthquake_agent = create_agent(
            self.model,
            tools=[*tools],
            response_format=ThreatAssessment,
            system_prompt=earthquake_prompt.format(
                format_name=ThreatAssessment.__name__
            ),
            name="Earthquake Agent",
        )

        workflow = create_swarm(
            [coordinator, fire_agent, earthquake_agent],
            default_active_agent="Coordinator",
        )

        self.app = workflow.compile(debug=debug)
        if print_agent:
            logger.info("\n" + self.app.get_graph().draw_ascii())

        logger.info("Swarm graph compiled successfully.")

    async def process_data(self, data: dict) -> dict:
        if not self.app:
            await self.initialize_graph()

        thread_id = str(data.get("room", "default_room"))
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": self.callbacks,
        }

        logger.info(f"[Agent] -> Executing Swarm for room: {thread_id}")

        prompt = self.prompt_template.format(data=data)

        initial_state: AgentState = {"messages": [HumanMessage(content=prompt)]}

        # Swarm execution
        result = await self.app.ainvoke(initial_state, config=config)

        assessment_data = self._extract_assessment(result)

        if not assessment_data:
            # Safety fallback in case the LLM did not use the tool
            logger.warning("The ThreatAssessment tool was not called.")
            return {}

        # Native Pydantic Validation
        assessment = ThreatAssessment(**assessment_data)

        return assessment.model_dump()
