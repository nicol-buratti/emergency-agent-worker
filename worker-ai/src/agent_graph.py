import logging
import operator
from typing import Annotated, Any, Literal, TypedDict, Union, get_args

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError

from src.app_settings import AppSettings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
ExpertType = Literal["fire", "earthquake"]
VALID_EXPERTS = get_args(ExpertType)


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
        description="Severity score on a scale from 0.0 to 1.0.", ge=0.0, le=1.0
    )
    justification: str = Field(
        description="Brief justification for the assigned danger level."
    )


class EscalateAction(BaseModel):
    action: Literal["escalate"] = Field(
        description="Select this action if a specialized analysis is required."
    )
    required_experts: list[ExpertType] = Field(
        description="Experts required. Do not generate assessment."
    )


class AssessAction(BaseModel):
    action: Literal["assess"] = Field(
        description="Select this action if the conditions are safe and no experts are needed."
    )
    assessments: list[ThreatAssessment] = Field(
        description="List of safety assessments."
    )


TriageOutput = Union[EscalateAction, AssessAction]


class ExpertOutput(BaseModel):
    assessments: list[ThreatAssessment] = Field(
        description="List of threat assessments for the primary room and any affected neighboring rooms."
    )


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    data: dict[str, Any]
    required_experts: list[str]
    assessments: Annotated[list[dict[str, Any]], operator.add]


class ExpertState(TypedDict):
    messages: list[AnyMessage]
    data: dict[str, Any]


class HazardMapReduceManager:
    def __init__(self) -> None:
        # Pydantic parses .env and environment variables here
        settings = AppSettings()

        extra_body: dict[str, Any] | None = (
            {"models": settings.extra_llm_models} if settings.extra_llm_models else None
        )

        self.model: ChatOpenAI = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            temperature=0.2,
            max_retries=2,
            extra_body=extra_body,
        )

        self.callbacks: list[Any] = []
        template_string: str = """IoT Context & Input Data:
{data}

Instructions:
Evaluate the reading values for safety/threat levels. Assess the primary room and determine if threat propagation requires assessing neighboring rooms."""
        self.prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["data"], template=template_string
        )
        self.app: Any | None = None

    async def initialize_graph(
        self,
        tools: list[Any] | None = None,
        debug: bool = False,
        print_agent: bool = False,
    ) -> None:
        tools = tools or []

        triage_sys: SystemMessage = SystemMessage(
            content="""
You are the Hazard Assessment Triage.
Triage the provided IoT data to assess room safety.
When necessary, use the provided Memgraph database tools to query the building's topology and inspect node parameters to understand the spatial layout and current sensor snapshot.
Transfer control to the specialized Agents if anomalies exist.
If both the telemetry data and the building's node parameters indicate normal, safe conditions, terminate directly with a safe assessment
            """
        )
        fire_sys: SystemMessage = SystemMessage(
            content="""
You are the Fire Safety Expert.
Analyze telemetry for fire hazards (heat, smoke, CO2 spikes).
If anomalous data is detected, use the Memgraph database tools to check the building's topology and node parameters.
You must identify adjacent rooms, ventilation paths, or connected structural nodes to assess the risk of fire and smoke spread.
Formulate a clear diagnosis based on both the telemetry and the building's spatial graph, and terminate the swarm once ready
        """
        )
        earthquake_sys: SystemMessage = SystemMessage(
            content="""
You are the Earthquake Safety Expert.
Analyze telemetry for seismic activity (vibrations, acceleration, structural shifts).
When assessing seismic impact, use the Memgraph database tools to query the building's topology and structural node parameters.
You must understand load-bearing dependencies, material parameters, and damage propagation across connected building elements.
Formulate a clear diagnosis based on the telemetry and the building's structural graph, and terminate the swarm once ready.
"""
        )

        triage_agent = create_agent(
            self.model,
            tools=tools,
            response_format=TriageOutput,
            system_prompt=triage_sys,
            name="Triage",
        )
        fire_agent = create_agent(
            self.model,
            tools=tools,
            response_format=ExpertOutput,
            system_prompt=fire_sys,
            name="Fire Agent",
        )
        earthquake_agent = create_agent(
            self.model,
            tools=tools,
            response_format=ExpertOutput,
            system_prompt=earthquake_sys,
            name="Earthquake Agent",
        )

        async def run_triage(state: GraphState) -> dict[str, Any]:
            invocation_result: dict[str, Any] = await triage_agent.ainvoke(
                {"messages": self.prompt_template.format(data=state["data"])}
            )
            triage_result: TriageOutput = invocation_result.get("structured_response")

            if triage_result.action == "escalate":
                return {
                    "required_experts": triage_result.required_experts,
                    "assessments": [],
                }

            return {
                "required_experts": [],
                "assessments": [a.model_dump() for a in triage_result.assessments],
            }

        def route_experts(state: GraphState) -> list[Send]:
            experts: list[str] = state.get("required_experts", [])
            if not experts:
                return [Send("safe_node", {})]

            sends: list[Send] = []
            input_prompt: HumanMessage = HumanMessage(
                content=self.prompt_template.format(data=state["data"])
            )

            for expert in experts:
                expert_lower = expert.lower()
                if expert_lower in VALID_EXPERTS:
                    sends.append(
                        Send(
                            f"{expert_lower}_node",
                            {
                                "messages": [input_prompt],
                                # "data": state["data"],
                            },
                        )
                    )
            return sends

        async def run_safe_node(state: GraphState) -> dict[str, Any]:
            return {}

        async def run_fire(state: ExpertState) -> dict[str, list[dict[str, Any]]]:
            invocation_result: dict[str, Any] = await fire_agent.ainvoke(state)
            expert_result: ExpertOutput = invocation_result.get("structured_response")
            return {"assessments": [a.model_dump() for a in expert_result.assessments]}

        async def run_earthquake(state: ExpertState) -> dict[str, list[dict[str, Any]]]:
            invocation_result: dict[str, Any] = await earthquake_agent.ainvoke(state)
            expert_result: ExpertOutput = invocation_result.get("structured_response")
            return {"assessments": [a.model_dump() for a in expert_result.assessments]}

        builder = StateGraph(GraphState)
        builder.add_node("triage", run_triage)
        builder.add_node("fire_node", run_fire)
        builder.add_node("earthquake_node", run_earthquake)
        builder.add_node("safe_node", run_safe_node)

        builder.add_edge(START, "triage")

        # Dynamically construct the allowed nodes list
        allowed_nodes = [f"{expert}_node" for expert in VALID_EXPERTS] + ["safe_node"]
        builder.add_conditional_edges("triage", route_experts, allowed_nodes)

        builder.add_edge("fire_node", END)
        builder.add_edge("earthquake_node", END)
        builder.add_edge("safe_node", END)

        self.app = builder.compile(debug=debug)

        if print_agent and self.app:
            logger.info("\n" + self.app.get_graph().draw_ascii())

    async def process_data(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.app:
            await self.initialize_graph()

        thread_id: str = str(data.get("room", "default_room"))
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "callbacks": self.callbacks,
        }

        initial_state: dict[str, Any] = {
            "messages": [],
            "data": data,
            "required_experts": [],
            "assessments": [],
        }

        graph_result: dict[str, Any] = await self.app.ainvoke(
            initial_state, config=config
        )
        raw_assessments: list[dict[str, Any]] = graph_result.get("assessments", [])

        validated_assessments: list[dict[str, Any]] = []
        for assessment_data in raw_assessments:
            try:
                assessment = ThreatAssessment(**assessment_data)
                validated_assessments.append(assessment.model_dump())
            except ValidationError as e:
                logger.error(
                    f"Validation failed for assessment data {assessment_data}: {e}"
                )

        return validated_assessments
