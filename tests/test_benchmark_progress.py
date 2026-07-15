"""Terminal and redirected-output behavior for benchmark progress rendering."""

from io import StringIO
from pathlib import Path

from data_analysis_agent.benchmark import (
    DEFAULT_TASKS_ROOT,
    _offline_model_factory,
    run_benchmark,
)
from data_analysis_agent.benchmark_progress import BenchmarkProgressRenderer
from data_analysis_agent.benchmark_types import BenchmarkConfig


def _renderer(*, interactive: bool, artifact_path: Path | None = None):
    stream = StringIO()
    return BenchmarkProgressRenderer(
        stream=stream,
        interactive=interactive,
        artifact_path=artifact_path,
    ), stream


def _start(renderer: BenchmarkProgressRenderer) -> None:
    renderer.emit(
        {
            "type": "benchmark_started",
            "repeat_index": 1,
            "repeats": 1,
            "task_id": "example_task",
            "approach": "agent",
            "model": "example-model",
        }
    )


def _three_goal_plan(renderer: BenchmarkProgressRenderer) -> None:
    renderer.emit(
        {
            "type": "plan_available",
            "goals": [
                {"goal_id": "G1", "objective": "Load inputs"},
                {"goal_id": "G2", "objective": "Filter records"},
                {"goal_id": "G3", "objective": "Compute result"},
            ],
        }
    )


def test_noninteractive_progress_shows_goals_after_verifier_approval(
    tmp_path: Path,
) -> None:
    renderer, stream = _renderer(
        interactive=False, artifact_path=tmp_path / "progress_events.jsonl"
    )
    _start(renderer)
    _three_goal_plan(renderer)
    renderer.emit({"type": "goal_started", "goal_id": "G1", "objective": "Load inputs"})
    renderer.emit({"type": "activity", "message": "Code execution — success"})

    before_pass = stream.getvalue()
    assert before_pass.count("BENCHMARK RUN 1/1") == 1
    assert "Planner proposed 3 steps" in before_pass
    assert "Progress: [0/3]" in before_pass
    assert "→ G1 — Load inputs" in before_pass
    assert "✓ G1 — Load inputs" not in before_pass
    assert "\x1b" not in before_pass

    renderer.emit({"type": "activity", "message": "Verifier — PASS"})
    renderer.emit({"type": "goal_completed", "goal_id": "G1"})
    renderer.emit(
        {
            "type": "goal_started",
            "goal_id": "G2",
            "objective": "Filter records",
        }
    )
    output = stream.getvalue()

    assert "✓ G1 — Load inputs" in output
    assert "Progress: [1/3]" in output
    assert "Current step: G2 — Filter records" in output
    assert (tmp_path / "progress_events.jsonl").is_file()


def test_progress_distinguishes_repair_and_replan_and_sanitizes_errors() -> None:
    renderer, stream = _renderer(interactive=False)
    _start(renderer)
    renderer.emit({"type": "activity", "message": "Mechanical repair: [2/50]"})
    renderer.emit({"type": "activity", "message": "Planner repair: [1/2]"})
    renderer.emit({"type": "activity", "message": "Scientific replan: [1/1]"})
    renderer.emit({"type": "error", "error": "\x1b[31mRuntimeError\x07: bad\nvalue"})
    output = stream.getvalue()

    assert "Mechanical repair: [2/50]" in output
    assert "Planner repair: [1/2]" in output
    assert "Scientific replan: [1/1]" in output
    assert "RuntimeError : bad value" in output
    assert "\x1b" not in output


def test_interactive_progress_redraws_current_step() -> None:
    renderer, stream = _renderer(interactive=True)
    _start(renderer)
    _three_goal_plan(renderer)
    renderer.emit({"type": "goal_started", "goal_id": "G1", "objective": "Load inputs"})
    renderer.emit({"type": "goal_completed", "goal_id": "G1"})
    renderer.emit(
        {
            "type": "goal_started",
            "goal_id": "G2",
            "objective": "Filter records",
        }
    )

    output = stream.getvalue()
    assert "\x1b[2J\x1b[H" in output
    assert output.endswith("Current step: G2 — Filter records\n" + "-" * 60 + "\n")


def test_agent_progress_stream_uses_approved_plan_without_ansi(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        model="offline-scripted-smoke",
        task_ids=["successive_difference_smoke"],
        approaches=["agent"],
        live=True,
    )
    stream = StringIO()

    run_benchmark(
        config=config,
        tasks_root=DEFAULT_TASKS_ROOT,
        output_root=tmp_path / "runs",
        model_factory=_offline_model_factory,
        project_root=tmp_path,
        progress_stream=stream,
        progress_interactive=False,
    )

    output = stream.getvalue()
    assert "Planner proposed 1 steps" in output
    assert "Progress: [0/1]" in output
    assert "✓ compute_successive_difference" in output
    assert "Progress: [1/1]" in output
    assert "\x1b" not in output
