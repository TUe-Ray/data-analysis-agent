import json

import pytest
from pydantic import ValidationError

from data_analysis_agent.final_output import (
    DeterministicFinalOutputProvider,
    FinalGenerationRequest,
    ScriptedFinalOutputProvider,
    build_scripted_output_provider,
)
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import build_scripted_model
from data_analysis_agent.schemas import FinalAnswer
from data_analysis_agent.state import AgentState


def initial_state() -> AgentState:
    return {
        "question": "Calculate mean, sample standard error, and count.",
        "file_paths": ["measurements.csv"],
        "input_context": "File: measurements.csv\nvalue\n10\n12\n14\n16",
        "replan_count": 0,
        "max_replans": 1,
        "output_repair_count": 0,
        "max_output_repairs": 1,
        "output_validation_history": [],
        "trace": [],
    }


def test_final_answer_accepts_valid_json_safe_object() -> None:
    answer = FinalAnswer.model_validate(
        {
            "status": "completed",
            "answer": "Mean is 13.",
            "key_results": {"mean": 13.0, "labels": ["approved", None]},
            "limitations": [],
        }
    )

    assert answer.key_results["mean"] == 13.0


def test_deterministic_generator_extracts_explicit_live_style_labels() -> None:
    provider = DeterministicFinalOutputProvider()
    raw = provider.generate(
        FinalGenerationRequest(
            question="Calculate requested summaries.",
            approved_execution_result=(
                "Mean\u202f=\u202f13\n"
                "Standard error\u202f≈\u202f1.291\n"
                "Observations\u202f=\u202f4"
            ),
            verifier_decision="PASS",
            verifier_feedback="Approved.",
            iteration_history=[],
        )
    )

    answer = FinalAnswer.model_validate_json(raw)
    assert answer.key_results == {
        "mean": 13.0,
        "sample_standard_error": 1.291,
        "n_observations": 4,
    }


@pytest.mark.parametrize(
    "value",
    [
        {
            "status": "completed",
            "answer": "Mean is 13.",
            "key_results": {"mean": 13},
        },
        {
            "status": "success",
            "answer": "Mean is 13.",
            "key_results": {"mean": 13},
            "limitations": [],
        },
    ],
)
def test_final_answer_rejects_missing_field_or_invalid_status(
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        FinalAnswer.model_validate(value)


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        'Final answer:\n{"status":"completed"}',
        '```json\n{"status":"completed"}\n```',
    ],
)
def test_non_json_or_extra_prose_is_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        FinalAnswer.model_validate_json(raw)


def test_valid_json_routes_directly_to_end() -> None:
    result = build_graph(
        build_scripted_model("valid-json"),
        build_scripted_output_provider("valid-json"),
    ).invoke(initial_state())

    assert result["output_validation_status"] == "VALID"
    assert result["output_repair_count"] == 0
    assert result["trace"][-2:] == [
        "final_answer_generator",
        "output_validator:VALID",
    ]
    assert json.loads(result["final_answer"])["status"] == "completed"


def test_invalid_json_repairs_once_and_routes_to_end() -> None:
    provider = build_scripted_output_provider("output-repair")
    result = build_graph(build_scripted_model("output-repair"), provider).invoke(
        initial_state()
    )

    assert result["status"] == "completed"
    assert result["output_repair_count"] == 1
    assert result["trace"][-4:] == [
        "final_answer_generator",
        "output_validator:INVALID",
        "output_repair",
        "output_validator:VALID",
    ]
    assert [item["status"] for item in result["output_validation_history"]] == [
        "INVALID",
        "VALID",
    ]
    assert '"limitations"' not in result["raw_final_output"]
    assert '"limitations"' in result["raw_repair_output"]
    FinalAnswer.model_validate_json(result["final_answer"])


def test_second_invalid_output_terminates_without_claiming_completion() -> None:
    result = build_graph(
        build_scripted_model("output-failure"),
        build_scripted_output_provider("output-failure"),
    ).invoke(initial_state())

    assert result["output_repair_count"] == 1
    assert result["output_validation_status"] == "INVALID"
    assert result["status"] == "output_validation_failed"
    assert result["trace"][-1] == "output_failure"
    failure = json.loads(result["final_answer"])
    assert failure["status"] == "output_validation_failed"
    assert failure["answer"] is None


def test_repair_receives_only_bounded_formatting_context() -> None:
    provider = ScriptedFinalOutputProvider(
        candidates=['{"status":"completed"}'],
        repairs=[
            json.dumps(
                {
                    "status": "completed",
                    "answer": "Mean = 13.",
                    "key_results": {"mean": 13},
                    "limitations": [],
                }
            )
        ],
    )
    build_graph(build_scripted_model("valid-json"), provider).invoke(initial_state())

    assert len(provider.repair_requests) == 1
    request = provider.repair_requests[0]
    assert set(vars(request)) == {
        "invalid_raw_output",
        "validation_error",
        "required_schema",
        "approved_execution_result",
    }
    assert "conversation" not in vars(request)
    assert "messages" not in vars(request)
