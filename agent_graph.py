import json
import logging
import operator
import os
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langfuse.langchain import CallbackHandler

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

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
    assessment: ThreatAssessment | None = Field(
        description="Threat assessment if no experts are required. Null if routing to experts."
    )


class GraphState(TypedDict):
    messages: Annotated[list, operator.add]
    data: dict
    required_experts: list[str]
    assessments: Annotated[list[dict], operator.add]


class ExpertState(TypedDict):
    messages: list
    data: dict


class HazardMapReduceManager:
    def __init__(self):
        extra_models_env = os.getenv("EXTRA_LLM_MODELS")
        extra_body = (
            {"models": json.loads(extra_models_env)} if extra_models_env else None
        )

        self.model = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0.2,
            max_retries=2,
            extra_body=extra_body,
        )
        self.callbacks = []
        if os.getenv("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "t", "yes"):
            self.callbacks.append(CallbackHandler())

        template_string = """IoT Context & Input Data:
{data}

Instructions:
Evaluate the reading values for safety/threat levels."""
        self.prompt_template = PromptTemplate(
            input_variables=["data"], template=template_string
        )
        self.app = None

    async def initialize_graph(
        self, tools: list = None, debug: bool = False, print_agent: bool = False
    ) -> None:

        # Enforce direct structured output for speed (replaces slow agent loops)
        triage_llm = self.model.with_structured_output(TriageOutput)
        expert_llm = self.model.with_structured_output(ThreatAssessment)

        triage_sys = SystemMessage(
            content="You are the Hazard Assessment Triage. Determine if 'fire' or 'earthquake' experts are required based on anomalies. If conditions are safe, provide the ThreatAssessment directly. Do not route if safe."
        )
        fire_sys = SystemMessage(
            content="You are the Fire Safety Expert. Analyze telemetry for fire hazards (heat, smoke, CO2 spikes). Provide a final ThreatAssessment."
        )
        earthquake_sys = SystemMessage(
            content="You are the Earthquake Safety Expert. Analyze telemetry for seismic activity. Provide a final ThreatAssessment."
        )

        async def run_triage(state: GraphState):
            input_messages = [
                triage_sys,
                HumanMessage(content=self.prompt_template.format(data=state["data"])),
            ]
            result = await triage_llm.ainvoke(input_messages)

            assessments = (
                [result.assessment.model_dump()]
                if not result.route_to_experts and result.assessment
                else []
            )
            return {
                "required_experts": result.required_experts,
                "assessments": assessments,
            }

        def route_experts(state: GraphState):
            experts = state.get("required_experts", [])
            if not experts:
                return [Send("safe_node", {})]

            sends = []
            # Isolate input: send only initial prompt and data, avoiding Triage contamination
            input_prompt = HumanMessage(
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

        async def run_safe_node(state: dict):
            return {}

        async def run_fire(state: ExpertState):
            result = await expert_llm.ainvoke(state["messages"])
            return {"assessments": [result.model_dump()]}

        async def run_earthquake(state: ExpertState):
            result = await expert_llm.ainvoke(state["messages"])
            return {"assessments": [result.model_dump()]}

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

        if print_agent:
            logger.info("\n" + self.app.get_graph().draw_ascii())

    async def process_data(self, data: dict) -> list[dict]:
        if not self.app:
            await self.initialize_graph()

        thread_id = str(data.get("room", "default_room"))
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": self.callbacks,
        }

        initial_state = {
            "messages": [],
            "data": data,
            "required_experts": [],
            "assessments": [],
        }

        result = await self.app.ainvoke(initial_state, config=config)
        raw_assessments = result.get("assessments", [])

        validated_assessments = []
        for assessment_data in raw_assessments:
            try:
                assessment = ThreatAssessment(**assessment_data)
                validated_assessments.append(assessment.model_dump())
            except Exception as e:
                logger.error(
                    f"Validation failed for assessment data {assessment_data}: {e}"
                )

        return validated_assessments
