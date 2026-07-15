from collections import Counter
from pathlib import Path

from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel, build_scripted_model
from data_analysis_agent.state import AgentState


def initial_state(*, trap: bool = False) -> AgentState:
    question = (
        "Calculate mean, sample standard error, and count."
        if trap
        else "Calculate mean and count."
    )
    return {
        "question": question,
        "file_paths": ["measurements.csv"],
        "input_context": "File: measurements.csv\nvalue\n10\n12\n14\n16",
        "replan_count": 0,
        "max_replans": 1,
        "trace": [],
    }


def role_counts(model: ScriptedRoleModel) -> Counter[str]:
    return Counter(call.role for call in model.calls)


def test_graph_compiles_successfully() -> None:
    graph = build_graph(build_scripted_model("happy"))

    assert {
        "planner",
        "executor",
        "verifier",
        "final_answer_generator",
        "output_validator",
        "output_repair",
        "failure_finalizer",
        "output_failure",
    }.issubset(graph.get_graph().nodes)


def test_happy_path_reaches_finalize_with_pass(tmp_path: Path) -> None:
    model = build_scripted_model("happy")
    result = build_graph(model).invoke(initial_state())

    assert result["verification_decision"] == "PASS"
    assert result["status"] == "completed"
    assert result["replan_count"] == 0
    assert result["trace"] == [
        "planner",
        "executor",
        "verifier:PASS",
        "final_answer_generator",
        "output_validator:VALID",
    ]
    assert role_counts(model) == {"planner": 1, "executor": 1, "verifier": 1}
    assert not Path(result["run_directory"]).exists()
    assert not (tmp_path / "runs").exists()


def test_replan_routes_back_and_recovers() -> None:
    model = build_scripted_model("replan")
    result = build_graph(model).invoke(initial_state(trap=True))

    assert result["status"] == "completed"
    assert result["verification_decision"] == "PASS"
    assert result["replan_count"] == 1
    assert result["trace"] == [
        "planner",
        "executor",
        "verifier:REPLAN",
        "planner",
        "executor",
        "verifier:PASS",
        "final_answer_generator",
        "output_validator:VALID",
    ]
    assert role_counts(model) == {"planner": 2, "executor": 2, "verifier": 2}
    assert result["iteration_history"] == [
        {
            "iteration": 1,
            "plan": (
                "1. Identify non-missing values.\n"
                "2. Calculate the arithmetic mean.\n"
                "3. Report the mean."
            ),
            "execution_result": "Mean = 13.",
            "verification_decision": "REPLAN",
            "verification_feedback": (
                "The user also requested the sample standard error and the number "
                "of observations used."
            ),
            "route": "Verifier -> Planner",
        },
        {
            "iteration": 2,
            "plan": result["plan"],
            "execution_result": result["execution_result"],
            "verification_decision": "PASS",
            "verification_feedback": result["verification_feedback"],
            "route": "Verifier -> Final Answer Generator",
        },
    ]


def test_max_replan_terminates_without_claiming_pass() -> None:
    model = build_scripted_model("max-replan")
    result = build_graph(model).invoke(initial_state(trap=True))

    assert result["status"] == "stopped_after_max_replans"
    assert result["verification_decision"] == "REPLAN"
    assert result["replan_count"] == 1
    assert result["trace"] == [
        "planner",
        "executor",
        "verifier:REPLAN",
        "planner",
        "executor",
        "verifier:REPLAN",
        "failure_finalizer:max_replans",
    ]
    assert "Verification did not pass" in result["final_answer"]
    assert role_counts(model) == {"planner": 2, "executor": 2, "verifier": 2}


def test_trap_recovery_final_answer_contains_all_requested_values() -> None:
    result = build_graph(build_scripted_model("replan")).invoke(
        initial_state(trap=True)
    )

    assert "Mean = 13" in result["final_answer"]
    assert "Sample standard error = 1.291" in result["final_answer"]
    assert "Number of observations used = 4" in result["final_answer"]
