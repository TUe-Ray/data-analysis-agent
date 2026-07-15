from pathlib import Path
from unittest.mock import patch

import pytest

from data_analysis_agent.demo import (
    PROJECT_ROOT,
    DemoInputError,
    format_workflow_result,
    main,
    run_demo,
    stage_input_files,
)


def test_stage_input_files_includes_name_and_content() -> None:
    path = PROJECT_ROOT / "examples/data/simple_measurements.csv"

    names, context = stage_input_files([path])

    assert names == ["simple_measurements.csv"]
    assert "File: simple_measurements.csv" in context
    assert "s4,16" in context


def test_missing_example_file_produces_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(DemoInputError, match="does not exist"):
        stage_input_files([missing])


@pytest.mark.parametrize("scenario", ["happy", "replan", "max-replan"])
def test_offline_demo_does_not_load_or_construct_live_client(scenario: str) -> None:
    with (
        patch("data_analysis_agent.demo.load_settings") as load_settings,
        patch("data_analysis_agent.demo.create_nebius_client") as create_client,
    ):
        result = run_demo(mode="offline", scenario=scenario)

    load_settings.assert_not_called()
    create_client.assert_not_called()
    assert result["status"] in {"completed", "stopped_after_max_replans"}


def test_replan_demo_returns_all_three_requested_values() -> None:
    result = run_demo(mode="offline", scenario="replan")

    assert result["status"] == "completed"
    assert "Mean = 13" in result["final_answer"]
    assert "Sample standard error = 1.291" in result["final_answer"]
    assert "Number of observations used = 4" in result["final_answer"]
    assert len(result["iteration_history"]) == 2
    assert result["iteration_history"][0]["verification_decision"] == "REPLAN"
    assert result["iteration_history"][1]["verification_decision"] == "PASS"


def test_offline_cli_prints_public_graph_state(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    exit_status = main(
        [
            "--mode",
            "offline",
            "--scenario",
            "happy",
            "--output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_status == 0
    for label in (
        "PROTOTYPE V0 — HAPPY DEMO",
        "Question:",
        "Staged files:",
        "ITERATION 1",
        "Plan:",
        "Execution result:",
        "Decision : PASS",
        "Route:",
        "FINAL RESULT",
        "Global replans            : 0",
        "JSON:",
        "Detailed log:",
    ):
        assert label in output


def test_replan_terminal_output_shows_both_iterations(tmp_path: Path) -> None:
    result = run_demo(mode="offline", scenario="replan")

    output = format_workflow_result(
        mode="offline",
        scenario="replan",
        result=result,
        log_path=tmp_path / "workflow.log",
    )

    assert "ITERATION 1" in output
    assert "ITERATION 2" in output
    assert "Decision : REPLAN" in output
    assert "Verifier -> Planner" in output
    assert "Decision : PASS" in output
    assert "Verifier -> Final Answer Generator" in output


@pytest.mark.parametrize(
    ("scenario", "expected_status", "repair_count"),
    [
        ("valid-json", "completed", 0),
        ("output-repair", "completed", 1),
        ("malformed-json", "completed", 1),
        ("output-failure", "output_validation_failed", 1),
    ],
)
def test_output_validation_demo_scenarios(
    scenario: str, expected_status: str, repair_count: int
) -> None:
    with (
        patch("data_analysis_agent.demo.load_settings") as load_settings,
        patch("data_analysis_agent.demo.create_nebius_client") as create_client,
    ):
        result = run_demo(mode="offline", scenario=scenario)

    load_settings.assert_not_called()
    create_client.assert_not_called()
    assert result["status"] == expected_status
    assert result["output_repair_count"] == repair_count


def test_output_repair_terminal_is_readable(tmp_path: Path) -> None:
    result = run_demo(mode="offline", scenario="output-repair")

    output = format_workflow_result(
        mode="offline",
        scenario="output-repair",
        result=result,
        log_path=tmp_path / "workflow.log",
    )

    for text in (
        "FINAL ANSWER",
        "Attempt 1 : INVALID",
        "Output Validator -> Output Repair",
        "Repair attempt : 1",
        "Attempt 2 : VALID",
        "Status                    : completed",
    ):
        assert text in output


def test_generated_python_terminal_uses_compact_artifact_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "run" / "workflow.log"
    log_path.parent.mkdir()
    result = run_demo(
        mode="offline", scenario="generated-python-repair", log_path=log_path
    )
    monkeypatch.setattr("data_analysis_agent.demo.PROJECT_ROOT", tmp_path)

    output = format_workflow_result(
        mode="offline",
        scenario="generated-python-repair",
        result=result,
        log_path=log_path,
    )

    artifact_path = result["completed_goal_results"][0]["artifact_paths"][0]
    assert "artifact_paths" not in output
    assert artifact_path not in output
    assert "Artifacts     : " in output
    assert "files saved" in output
    assert "Artifact directory:\nrun/goals/compute_successive_difference/" in output
    assert "Generated script versions : 2" in output
    detailed_log = log_path.read_text(encoding="utf-8")
    assert artifact_path in detailed_log
    assert '"artifact_paths"' in detailed_log
