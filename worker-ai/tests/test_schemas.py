import pytest
from pydantic import ValidationError

from src.agent_graph import (
    AssessAction,
    EscalateAction,
    ExpertOutput,
    ThreatAssessment,
    VALID_EXPERTS,
)


class TestThreatAssessment:
    def test_accepts_valid_payload(self, valid_assessment_payload):
        assessment = ThreatAssessment(**valid_assessment_payload)

        assert assessment.room == "room-101"
        assert assessment.danger == "none"
        assert assessment.danger_score == 0.0

    @pytest.mark.parametrize(
        "field, bad_value",
        [
            ("warning", "critical"),  # not in {none, pre-alert}
            ("danger", "extreme"),  # not in {high, medium, low, none}
            ("danger_type", "flood"),  # not in the fixed enum
        ],
    )
    def test_rejects_invalid_enum_values(
        self, valid_assessment_payload, field, bad_value
    ):
        payload = {**valid_assessment_payload, field: bad_value}

        with pytest.raises(ValidationError):
            ThreatAssessment(**payload)

    @pytest.mark.parametrize("score", [-0.1, 1.1, 2.0, -5])
    def test_rejects_danger_score_outside_zero_one_range(
        self, valid_assessment_payload, score
    ):
        payload = {**valid_assessment_payload, "danger_score": score}

        with pytest.raises(ValidationError):
            ThreatAssessment(**payload)

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_accepts_danger_score_at_and_within_bounds(
        self, valid_assessment_payload, score
    ):
        payload = {**valid_assessment_payload, "danger_score": score}

        assessment = ThreatAssessment(**payload)

        assert assessment.danger_score == score

    def test_rejects_missing_required_field(self, valid_assessment_payload):
        payload = dict(valid_assessment_payload)
        del payload["justification"]

        with pytest.raises(ValidationError):
            ThreatAssessment(**payload)

    def test_rejects_non_numeric_danger_score(self, valid_assessment_payload):
        """Guards against a malformed LLM response that puts a string like
        'high' into a numeric field instead of a real score."""
        payload = {**valid_assessment_payload, "danger_score": "high"}

        with pytest.raises(ValidationError):
            ThreatAssessment(**payload)

    def test_extra_unexpected_fields_do_not_crash_validation(
        self, valid_assessment_payload
    ):
        """Default pydantic behavior ignores unknown fields; assert this
        explicitly since a stray field from the LLM shouldn't take down
        the whole assessment."""
        payload = {**valid_assessment_payload, "confidence": 0.9}

        assessment = ThreatAssessment(**payload)

        assert not hasattr(assessment, "confidence")


class TestEscalateAction:
    def test_accepts_valid_experts(self):
        action = EscalateAction(action="escalate", required_experts=["fire"])

        assert action.action == "escalate"
        assert action.required_experts == ["fire"]

    def test_accepts_both_known_expert_types(self):
        action = EscalateAction(
            action="escalate", required_experts=["fire", "earthquake"]
        )

        assert set(action.required_experts) == {"fire", "earthquake"}

    def test_rejects_unknown_expert_type(self):
        with pytest.raises(ValidationError):
            EscalateAction(action="escalate", required_experts=["flood"])

    def test_rejects_wrong_action_literal(self):
        with pytest.raises(ValidationError):
            EscalateAction(action="assess", required_experts=["fire"])


class TestAssessAction:
    def test_accepts_valid_assessment_list(self, valid_assessment_payload):
        action = AssessAction(action="assess", assessments=[valid_assessment_payload])

        assert action.action == "assess"
        assert len(action.assessments) == 1
        assert isinstance(action.assessments[0], ThreatAssessment)

    def test_accepts_empty_assessment_list(self):
        action = AssessAction(action="assess", assessments=[])

        assert action.assessments == []

    def test_rejects_invalid_nested_assessment(self, valid_assessment_payload):
        bad_payload = {**valid_assessment_payload, "danger": "extreme"}

        with pytest.raises(ValidationError):
            AssessAction(action="assess", assessments=[bad_payload])


class TestExpertOutput:
    def test_accepts_multiple_assessments_for_propagation(
        self, valid_assessment_payload
    ):
        neighbor_payload = {**valid_assessment_payload, "room": "room-102"}

        output = ExpertOutput(assessments=[valid_assessment_payload, neighbor_payload])

        assert len(output.assessments) == 2
        assert {a.room for a in output.assessments} == {"room-101", "room-102"}


def test_valid_experts_only_contains_fire_and_earthquake():
    """Regression guard: if a new expert type is ever added to the
    ExpertType literal, this test forces a conscious update rather than
    the routing logic silently picking it up (or not) elsewhere."""
    assert set(VALID_EXPERTS) == {"fire", "earthquake"}
