"""Terminal and redirected-output behavior for benchmark progress rendering."""

import json
from io import StringIO
from pathlib import Path

from data_analysis_agent.benchmark import (
    DEFAULT_TASKS_ROOT,
    _offline_model_factory,
    run_benchmark,
)
from data_analysis_agent.benchmark_approaches import _model_observer, run_agent
from data_analysis_agent.benchmark_progress import BenchmarkProgressRenderer
from data_analysis_agent.benchmark_tasks import load_benchmark_task, stage_public_task
from data_analysis_agent.benchmark_types import BenchmarkConfig
from data_analysis_agent.models import ScriptedRoleModel


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


def test_planner_progress_announces_plan_generation_before_steps() -> None:
    events: list[dict[str, str]] = []
    observer = _model_observer(lambda event: events.append(event))

    assert observer is not None
    observer("start", "planner", 1, 0.0, None)
    observer("end", "planner", 1, 2.5, None)

    assert events == [
        {"type": "activity", "message": "Planner — started; generating plan steps..."},
        {"type": "activity", "message": "Planner — plan generated in 2.5s"},
    ]


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


def test_elapsed_timer_is_visible_and_freezes_when_run_finishes() -> None:
    now = [100.0]
    stream = StringIO()
    renderer = BenchmarkProgressRenderer(
        stream=stream,
        interactive=False,
        clock=lambda: now[0],
    )
    _start(renderer)
    now[0] = 125.0
    renderer.emit({"type": "activity", "message": "Still running"})

    assert "Elapsed: 00:00:25" in stream.getvalue()

    renderer.emit({"type": "benchmark_finished"})
    now[0] = 999.0
    renderer.emit({"type": "activity", "message": "After finish"})
    assert stream.getvalue().count("Elapsed: 00:00:25") >= 2
    assert "Elapsed: 00:14:59" not in stream.getvalue()
    renderer.close()


def test_interactive_redraw_includes_elapsed_timer() -> None:
    now = [10.0]
    stream = StringIO()
    renderer = BenchmarkProgressRenderer(
        stream=stream,
        interactive=True,
        clock=lambda: now[0],
    )
    _start(renderer)
    now[0] = 71.0
    renderer.emit({"type": "activity", "message": "Waiting on model"})

    assert "Elapsed: 00:01:01" in stream.getvalue()
    renderer.close()


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
    assert "Planner — started; generating plan steps..." in output
    assert "Planner — plan generated in" in output
    assert "Planner proposed 1 steps" in output
    assert "→ G1 — Compute the mean absolute successive difference" in output
    assert "Progress: [0/1]" in output
    assert "✓ G1 — Compute the mean absolute successive difference" in output
    assert "Progress: [1/1]" in output
    assert output.index("Planner — started") < output.index("Planner proposed 1 steps")
    assert "\x1b" not in output


def test_new_plan_replaces_stale_goals_completion_and_current_state() -> None:
    renderer, _ = _renderer(interactive=False)
    renderer.emit(
        {
            "type": "plan_available",
            "goals": [
                {"goal_id": "G1", "objective": "Old completed goal"},
                {"goal_id": "G2", "objective": "Old current goal"},
            ],
            "completed_goal_ids": ["G1"],
            "current_goal_id": "G2",
            "scientific_replan_count": 0,
        }
    )
    renderer.emit(
        {
            "type": "plan_available",
            "goals": [
                {"goal_id": "G2", "objective": "Revised current goal"},
                {"goal_id": "G3", "objective": "New goal"},
            ],
            "completed_goal_ids": [],
            "current_goal_id": "G2",
            "scientific_replan_count": 1,
        }
    )

    assert [goal["goal_id"] for goal in renderer.goals] == ["G2", "G3"]
    assert renderer.completed == set()
    assert renderer.current_goal == "G2"
    assert renderer.scientific_replan_count == 1
    lines = renderer._progress_lines()
    assert all("Old completed goal" not in line for line in lines)
    assert "→ G1 — Revised current goal" in lines
    assert sum(line.startswith("✓") for line in lines) == 0
    assert lines[-1] == "Progress: [0/2]"


def test_new_plan_keeps_only_authoritative_preserved_completion() -> None:
    renderer, _ = _renderer(interactive=False)
    renderer.completed = {"stale"}
    renderer.emit(
        {
            "type": "plan_available",
            "goals": [
                {"goal_id": "G1", "objective": "Preserved"},
                {"goal_id": "G3", "objective": "Current"},
            ],
            "completed_goal_ids": ["G1"],
            "current_goal_id": "G3",
            "scientific_replan_count": 1,
        }
    )

    lines = renderer._progress_lines()
    assert renderer.completed == {"G1"}
    assert sum(line.startswith("✓") for line in lines) == 1
    assert lines[-1] == "Progress: [1/2]"


def test_exhausted_replan_budget_never_renders_two_of_one(tmp_path: Path) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, "successive_difference_smoke")
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    plan = json.dumps(
        {
            "scientific_objective": "Return a value.",
            "goals": [
                {
                    "goal_id": "G1",
                    "objective": "Return a value.",
                    "required_outputs": ["value"],
                    "constraints": [],
                    "success_criteria": ["A value is returned."],
                    "depends_on": [],
                }
            ],
        }
    )
    strategy = json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "Use generated Python.",
        }
    )
    generation = json.dumps(
        {
            "kind": "python",
            "code": "__agent_result__ = {'value': 1}\n",
            "summary": "Return the value.",
        }
    )
    replan = '{"decision":"REPLAN","feedback":"Revise the method."}'
    model = ScriptedRoleModel(
        {
            "planner": [plan, plan],
            "executor": [strategy, generation, strategy, generation],
            "verifier": [replan, replan],
        }
    )
    renderer, stream = _renderer(interactive=False)

    run_agent(
        public=public,
        model=model,
        run_directory=attempt,
        config=BenchmarkConfig(
            model="offline",
            task_ids=["successive_difference_smoke"],
            approaches=["agent"],
        ),
        progress=renderer.emit,
    )

    output = stream.getvalue()
    assert "Scientific replan: [1/1]" in output
    assert "Scientific replan budget exhausted" in output
    assert "[2/1]" not in output
