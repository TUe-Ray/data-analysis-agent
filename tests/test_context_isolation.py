from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ModelCall, build_scripted_model
from data_analysis_agent.prompts import (
    EXECUTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)
from data_analysis_agent.state import AgentState


def calls_for_role(calls: list[ModelCall], role: str) -> list[ModelCall]:
    return [call for call in calls if call.role == role]


def test_each_role_receives_a_separately_constructed_context() -> None:
    model = build_scripted_model("replan")
    state: AgentState = {
        "question": "Calculate the mean, standard error, and count.",
        "file_paths": ["measurements.csv"],
        "input_context": "File: measurements.csv\nvalue\n10\n12\n14\n16",
        "replan_count": 0,
        "max_replans": 1,
        "trace": [],
    }

    build_graph(model).invoke(state)

    planner_calls = calls_for_role(model.calls, "planner")
    executor_calls = calls_for_role(model.calls, "executor")
    verifier_calls = calls_for_role(model.calls, "verifier")

    assert all(len(call.messages) == 2 for call in model.calls)
    assert planner_calls[0].messages[0]["content"] == PLANNER_SYSTEM_PROMPT
    assert "Verifier feedback:" not in planner_calls[0].messages[1]["content"]
    assert "Current replan count: 0" in planner_calls[0].messages[1]["content"]
    assert "Verifier feedback:" in planner_calls[1].messages[1]["content"]
    assert "sample standard error" in planner_calls[1].messages[1]["content"]
    assert "Current replan count: 1" in planner_calls[1].messages[1]["content"]

    for call in executor_calls:
        assert call.messages[0]["content"] == EXECUTOR_SYSTEM_PROMPT
        assert "Question:" in call.messages[1]["content"]
        assert "Input context:" in call.messages[1]["content"]
        assert "Plan:" in call.messages[1]["content"]
        assert "Verifier feedback:" not in call.messages[1]["content"]

    for call in verifier_calls:
        assert call.messages[0]["content"] == VERIFIER_SYSTEM_PROMPT
        verifier_context = call.messages[1]["content"]
        assert "Question:" in verifier_context
        assert "Input context:" in verifier_context
        assert "Plan:" in verifier_context
        assert "Execution result:" in verifier_context
        assert PLANNER_SYSTEM_PROMPT not in verifier_context
        assert EXECUTOR_SYSTEM_PROMPT not in verifier_context
