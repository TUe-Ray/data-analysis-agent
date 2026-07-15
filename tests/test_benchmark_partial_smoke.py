"""Tests for the official goal-limited benchmark smoke mode."""

from pathlib import Path

from data_analysis_agent.benchmark import DEFAULT_TASKS_ROOT, main, run_benchmark
from data_analysis_agent.benchmark_types import BenchmarkConfig


def test_partial_smoke_uses_agent_stack_and_skips_full_grading(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        model="offline-scripted-smoke",
        task_ids=["successive_difference_smoke"],
        approaches=["agent"],
        stop_after_goals=1,
    )

    summary, results = run_benchmark(
        config=config,
        tasks_root=DEFAULT_TASKS_ROOT,
        output_root=tmp_path,
    )

    assert summary.config.stop_after_goals == 1
    assert results[0].status == "completed"
    assert results[0].graded is False
    assert results[0].partial_run is True
    assert results[0].partial_run_reached is True
    partial_summary = (
        tmp_path
        / summary.benchmark_run_id
        / "agent/successive_difference_smoke/repeat_001/agent_run"
        / "partial_run_summary.json"
    )
    assert partial_summary.is_file()


def test_partial_smoke_cli_returns_nonzero_when_target_is_not_reached(
    tmp_path: Path,
) -> None:
    status = main(
        [
            "--task",
            "successive_difference_smoke",
            "--approaches",
            "agent",
            "--stop-after-goals",
            "2",
            "--output-root",
            str(tmp_path),
            "--no-live-progress",
        ]
    )

    assert status == 1
