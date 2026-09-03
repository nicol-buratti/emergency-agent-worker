"""Tests for HazardMapReduceManager's graph-building and routing logic.

The LLM itself is never called: `create_agent` is monkeypatched so each
node's `.ainvoke()` returns a controlled, deterministic structured
response. This lets us assert on *our* routing/aggregation code (does an
"escalate" triage result fan out to the right expert nodes? does a "safe"
result skip the experts? do invalid assessments get dropped?) without any
network access or LLM non-determinism.
"""

from unittest.mock import AsyncMock

import pytest

import src.agent_graph as agent_graph
from src.agent_graph import HazardMapReduceManager


class FakeAgent:
    """Stand-in for the object returned by langchain.agents.create_agent.

    `responses` is a list of dicts to return from successive ainvoke()
    calls (pop from the front), so a test can script multi-call scenarios
    if ever needed; most tests only need one.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def ainvoke(self, state):
        self.calls.append(state)
        return self._responses.pop(0)


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    return HazardMapReduceManager()


def patch_agents(
    monkeypatch, *, triage_response, fire_response=None, earthquake_response=None
):
    """Monkeypatch create_agent so triage/fire/earthquake nodes return the
    given canned structured_response objects instead of calling a real LLM.
    Agents are handed out in the same order agent_graph.py creates them:
    triage, fire, earthquake.
    """
    agents = [
        FakeAgent([{"structured_response": triage_response}]),
        FakeAgent([{"structured_response": fire_response}]),
        FakeAgent([{"structured_response": earthquake_response}]),
    ]
    agent_iter = iter(agents)

    def fake_create_agent(*args, **kwargs):
        return next(agent_iter)

    monkeypatch.setattr(agent_graph, "create_agent", fake_create_agent)
    return agents


class TestSafeTriageSkipsExperts:
    @pytest.mark.asyncio
    async def test_no_anomaly_produces_final_assessment_without_escalating(
        self, manager, monkeypatch, valid_assessment_payload
    ):
        triage_response = agent_graph.AssessAction(
            action="assess",
            assessments=[agent_graph.ThreatAssessment(**valid_assessment_payload)],
        )
        agents = patch_agents(monkeypatch, triage_response=triage_response)

        await manager.initialize_graph(tools=[])
        result = await manager.process_data(
            {"room": "room-101", "sensor_data": [{"temp": 21}]}
        )

        assert result == [valid_assessment_payload]
        # Only the triage agent should ever have been invoked.
        assert agents[1].calls == []
        assert agents[2].calls == []


class TestEscalationRouting:
    @pytest.mark.asyncio
    async def test_fire_escalation_invokes_only_fire_expert(
        self, manager, monkeypatch, valid_assessment_payload
    ):
        triage_response = agent_graph.EscalateAction(
            action="escalate", required_experts=["fire"]
        )
        fire_payload = {
            **valid_assessment_payload,
            "danger": "high",
            "danger_type": "fire",
            "danger_score": 0.9,
        }
        fire_response = agent_graph.ExpertOutput(
            assessments=[agent_graph.ThreatAssessment(**fire_payload)]
        )
        agents = patch_agents(
            monkeypatch, triage_response=triage_response, fire_response=fire_response
        )

        await manager.initialize_graph(tools=[])
        result = await manager.process_data(
            {"room": "room-101", "sensor_data": [{"temp": 90}]}
        )

        assert result == [fire_payload]
        assert len(agents[1].calls) == 1  # fire agent invoked
        assert agents[2].calls == []  # earthquake agent never invoked

    @pytest.mark.asyncio
    async def test_earthquake_escalation_invokes_only_earthquake_expert(
        self, manager, monkeypatch, valid_assessment_payload
    ):
        triage_response = agent_graph.EscalateAction(
            action="escalate", required_experts=["earthquake"]
        )
        eq_payload = {
            **valid_assessment_payload,
            "danger": "medium",
            "danger_type": "earthquake",
            "danger_score": 0.5,
        }
        eq_response = agent_graph.ExpertOutput(
            assessments=[agent_graph.ThreatAssessment(**eq_payload)]
        )
        agents = patch_agents(
            monkeypatch,
            triage_response=triage_response,
            earthquake_response=eq_response,
        )

        await manager.initialize_graph(tools=[])
        result = await manager.process_data(
            {"room": "room-101", "sensor_data": [{"vibration": 8.5}]}
        )

        assert result == [eq_payload]
        assert agents[1].calls == []  # fire agent never invoked

    @pytest.mark.asyncio
    async def test_both_experts_escalated_results_are_merged(
        self, manager, monkeypatch, valid_assessment_payload
    ):
        """Map-reduce: triage can request both experts; the final
        assessments list should contain both experts' findings."""
        triage_response = agent_graph.EscalateAction(
            action="escalate", required_experts=["fire", "earthquake"]
        )
        fire_payload = {
            **valid_assessment_payload,
            "room": "room-101",
            "danger_type": "fire",
        }
        eq_payload = {
            **valid_assessment_payload,
            "room": "room-102",
            "danger_type": "earthquake",
        }
        fire_response = agent_graph.ExpertOutput(
            assessments=[agent_graph.ThreatAssessment(**fire_payload)]
        )
        eq_response = agent_graph.ExpertOutput(
            assessments=[agent_graph.ThreatAssessment(**eq_payload)]
        )
        patch_agents(
            monkeypatch,
            triage_response=triage_response,
            fire_response=fire_response,
            earthquake_response=eq_response,
        )

        await manager.initialize_graph(tools=[])
        result = await manager.process_data(
            {"room": "room-101", "sensor_data": [{"temp": 90, "vibration": 8.5}]}
        )

        rooms_and_types = {(a["room"], a["danger_type"]) for a in result}
        assert rooms_and_types == {("room-101", "fire"), ("room-102", "earthquake")}
        assert len(result) == 2


class TestAssessmentValidationFiltering:
    @pytest.mark.asyncio
    async def test_invalid_assessment_from_graph_is_dropped_not_raised(
        self, manager, monkeypatch, valid_assessment_payload
    ):
        """process_data re-validates every assessment coming out of the
        graph. If the graph state somehow contains a malformed dict, it
        must be logged and skipped rather than raising and killing the
        worker for the whole room."""
        triage_response = agent_graph.AssessAction.model_construct(
            action="assess", assessments=[]
        )
        patch_agents(monkeypatch, triage_response=triage_response)

        await manager.initialize_graph(tools=[])

        # Directly exercise the post-graph validation step with a mix of
        # one valid and one invalid raw assessment dict, bypassing the
        # need to coerce an actually-invalid object through Pydantic
        # models upstream.
        manager.app.ainvoke = AsyncMock(
            return_value={
                "assessments": [
                    valid_assessment_payload,
                    {**valid_assessment_payload, "danger": "extreme"},
                ]
            }
        )

        result = await manager.process_data({"room": "room-101", "sensor_data": []})

        assert result == [valid_assessment_payload]


class TestGraphInitialization:
    @pytest.mark.asyncio
    async def test_graph_compiles_with_expected_nodes(self, manager, monkeypatch):
        patch_agents(
            monkeypatch,
            triage_response=agent_graph.AssessAction(action="assess", assessments=[]),
        )

        await manager.initialize_graph(tools=[])

        node_names = set(manager.app.get_graph().nodes.keys())
        assert {"triage", "fire_node", "earthquake_node", "safe_node"}.issubset(
            node_names
        )

    @pytest.mark.asyncio
    async def test_process_data_initializes_graph_if_not_already_built(
        self, manager, monkeypatch, valid_assessment_payload
    ):
        """process_data() is defensive: if initialize_graph() was never
        called, it should build a (default, no-tools) graph itself rather
        than raising an AttributeError on self.app."""
        patch_agents(
            monkeypatch,
            triage_response=agent_graph.AssessAction(
                action="assess",
                assessments=[agent_graph.ThreatAssessment(**valid_assessment_payload)],
            ),
        )

        assert manager.app is None

        result = await manager.process_data(
            {"room": "room-101", "sensor_data": [{"temp": 21}]}
        )

        assert manager.app is not None
        assert result == [valid_assessment_payload]
