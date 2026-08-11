import json
import logging
import operator
import os
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


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


class TriageOutput(BaseModel):
    route_to_experts: bool = Field(
        description="True if specialized analysis is needed, False if safe to assess directly."
    )
    required_experts: list[Literal["fire", "earthquake"]] = Field(
        description="List of experts required. Empty if none."
    )
    assessments: list[ThreatAssessment] | None = Field(
        description="List of threat assessments if no experts are required, covering the primary room and any affected neighbors. Null if routing to experts."
    )


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
        extra_models_env: str | None = os.getenv("EXTRA_LLM_MODELS")
        extra_body: dict[str, Any] | None = (
            {"models": json.loads(extra_models_env)} if extra_models_env else None
        )

        self.model: ChatOpenAI = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0.2,
            max_retries=2,
            extra_body=extra_body,
        )
        self.callbacks: list[Any] = []
        if os.getenv("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "t", "yes"):
            self.callbacks.append(CallbackHandler())

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
            content="You are the Hazard Assessment Triage. Determine if 'fire' or 'earthquake' experts are required based on anomalies. If conditions are safe, provide the ThreatAssessments directly (including neighbors if relevant). Do not route if safe."
        )
        fire_sys: SystemMessage = SystemMessage(
            content="You are the Fire Safety Expert. Analyze telemetry for fire hazards (heat, smoke, CO2 spikes). Provide final ThreatAssessments for the primary room and any adjacent rooms at risk."
        )
        earthquake_sys: SystemMessage = SystemMessage(
            content="You are the Earthquake Safety Expert. Analyze telemetry for seismic activity. Provide final ThreatAssessments for the primary room and affected structural zones."
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
            messages: list[AnyMessage] = [
                triage_sys,
                HumanMessage(content=self.prompt_template.format(data=state["data"])),
            ]

            invocation_result: dict[str, Any] = await triage_agent.ainvoke(
                {"messages": messages}
            )
            triage_result: TriageOutput = invocation_result.get("structured_response")

            print(f"\n\nTriage result: {triage_result}\n\n")

            raw_assessments: list[ThreatAssessment] = triage_result.assessments or []
            assessments: list[dict[str, Any]] = (
                [
                    a.model_dump() if hasattr(a, "model_dump") else a
                    for a in raw_assessments
                ]
                if not triage_result.route_to_experts and raw_assessments
                else []
            )

            return {
                "required_experts": triage_result.required_experts,
                "assessments": assessments,
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
                if expert.lower() == "fire":
                    sends.append(
                        Send(
                            "fire_node",
                            {
                                "messages": [fire_sys, input_prompt],
                                "data": state["data"],
                            },
                        )
                    )
                elif expert.lower() == "earthquake":
                    sends.append(
                        Send(
                            "earthquake_node",
                            {
                                "messages": [earthquake_sys, input_prompt],
                                "data": state["data"],
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
        builder.add_conditional_edges(
            "triage", route_experts, ["fire_node", "earthquake_node", "safe_node"]
        )
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
            except Exception as e:
                logger.error(
                    f"Validation failed for assessment data {assessment_data}: {e}"
                )

        return validated_assessments
