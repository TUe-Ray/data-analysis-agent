import pytest
from pydantic import ValidationError

from data_analysis_agent.graph import build_graph, route_after_verification
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.nodes import VerifierOutputError
from data_analysis_agent.schemas import VerificationOutput
from data_analysis_agent.state import AgentState


def test_verification_output_validates_json() -> None:
    output = VerificationOutput.model_validate_json(
        '{"decision":"PASS","feedback":"All requested values are present."}'
    )

    assert output.decision == "PASS"


@pytest.mark.parametrize(
    "raw_output",
    [
        "PASS",
        '{"decision":"MAYBE","feedback":"Uncertain."}',
        '{"decision":"PASS","feedback":"ok","unexpected":true}',
    ],
)
def test_verification_output_rejects_invalid_json(raw_output: str) -> None:
    with pytest.raises(ValidationError):
        VerificationOutput.model_validate_json(raw_output)


def test_invalid_verifier_json_fails_after_one_repair_attempt() -> None:
    model = ScriptedRoleModel(
        {
            "planner": ["1. Calculate the mean."],
            "executor": ["Mean = 13."],
            "verifier": ["not-json", '{"decision":"UNKNOWN","feedback":"x"}'],
        }
    )
    state: AgentState = {
        "question": "Calculate the mean.",
        "file_paths": ["values.csv"],
        "input_context": "File: values.csv\nvalue\n10\n16",
        "replan_count": 0,
        "max_replans": 1,
        "trace": [],
    }

    with pytest.raises(VerifierOutputError, match="after one repair attempt"):
        build_graph(model).invoke(state)

    assert [call.role for call in model.calls].count("verifier") == 2


def test_router_honors_pass_and_replan_limit() -> None:
    assert route_after_verification({"verification_decision": "PASS"}) == "finalize"
    assert (
        route_after_verification(
            {
                "verification_decision": "REPLAN",
                "replan_count": 0,
                "max_replans": 1,
            }
        )
        == "planner"
    )
    assert (
        route_after_verification(
            {
                "verification_decision": "REPLAN",
                "replan_count": 1,
                "max_replans": 1,
            }
        )
        == "finalize"
    )
