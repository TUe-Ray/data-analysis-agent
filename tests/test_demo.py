from pathlib import Path
from unittest.mock import patch

import pytest

from data_analysis_agent.demo import (
    PROJECT_ROOT,
    DemoInputError,
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


def test_offline_cli_prints_public_graph_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_status = main(["--mode", "offline", "--scenario", "happy"])

    output = capsys.readouterr().out
    assert exit_status == 0
    for label in (
        "Mode:",
        "Question:",
        "Staged files:",
        "Plan:",
        "Execution result:",
        "Verification decision:",
        "Verification feedback:",
        "Replan count:",
        "Trace:",
        "Final status:",
        "Final answer:",
    ):
        assert label in output
