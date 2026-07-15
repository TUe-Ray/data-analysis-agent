import json
from pathlib import Path
from unittest.mock import patch

import pytest

from data_analysis_agent.benchmark import (
    DEFAULT_TASKS_ROOT,
    _offline_model_factory,
    benchmark_scope_label,
    build_benchmark_run_id,
    format_benchmark_summary,
    run_benchmark,
)
from data_analysis_agent.benchmark_approaches import (
    build_direct_answer_messages,
    build_one_shot_code_messages,
    run_agent,
    run_direct_answer,
    run_one_shot_code,
)
from data_analysis_agent.benchmark_grading import (
    compare_values,
    grade_candidate,
    invalid_candidate_grade,
    numeric_match,
)
from data_analysis_agent.benchmark_tasks import (
    load_benchmark_task,
    stage_public_task,
)
from data_analysis_agent.benchmark_types import BenchmarkConfig, PublicTaskView
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.python_runner import LocalPythonRunner


@pytest.fixture
def task():
    return load_benchmark_task(DEFAULT_TASKS_ROOT, "successive_difference_smoke")


@pytest.fixture
def config() -> BenchmarkConfig:
    return BenchmarkConfig(
        model="offline-test-model",
        task_ids=["successive_difference_smoke"],
        approaches=["direct_answer", "one_shot_code", "agent"],
    )


def _final(value: float) -> str:
    return json.dumps(
        {
            "status": "completed",
            "answer": "done",
            "key_results": {"mean_absolute_successive_difference": value},
            "limitations": [],
        }
    )


def _three_file_public_task() -> PublicTaskView:
    contents = {
        "patients.csv": "patient_id\nP001\n",
        "visits.csv": "patient_id,value\nP001,10\n",
        "exclusions.csv": "patient_id,effective_date\n",
    }
    return PublicTaskView(
        task_id="three_file_test",
        prompt="Read every supplied public CSV.",
        data_files=list(contents),
        data_contents=contents,
        answer_schema={"type": "object"},
    )


def _read_all_files_code(attempt: Path, *, legacy_stdout: bool = False) -> str:
    patients_path = (attempt / "inputs/patients.csv").resolve()
    visits_path = (attempt / "inputs/visits.csv").resolve()
    exclusions_path = (attempt / "inputs/exclusions.csv").resolve()
    result_statement = (
        'print("{\\"status\\":\\"completed\\",\\"answer\\":\\"read\\",'
        '\\"key_results\\":{\\"files_read\\":3},\\"limitations\\":[]}")'
        if legacy_stdout
        else "__agent_result__ = {'files_read': 3}"
    )
    return f"""import csv

with open({str(patients_path)!r}, encoding="utf-8") as handle:
    patients = list(csv.DictReader(handle))
with open({str(visits_path)!r}, encoding="utf-8") as handle:
    visits = list(csv.DictReader(handle))
with open({str(exclusions_path)!r}, encoding="utf-8") as handle:
    exclusions = list(csv.DictReader(handle))
{result_statement}
"""


def test_public_task_view_excludes_private_grading_fields(task) -> None:
    dumped = task.public.model_dump_json()
    assert "grader_path" not in dumped
    assert "reference_path" not in dumped
    assert "absolute_tolerance" not in dumped
    assert set(PublicTaskView.model_fields).isdisjoint(
        {"grader_path", "reference_path", "reference"}
    )


def test_baseline_messages_contain_only_public_material(task, tmp_path: Path) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    hidden_reference = Path(task.private.reference_path).read_text(encoding="utf-8")
    direct = "\n".join(
        message["content"] for message in build_direct_answer_messages(public)
    )
    code = "\n".join(
        message["content"] for message in build_one_shot_code_messages(public)
    )

    assert public.prompt in direct
    assert "s5,16" in direct
    assert public.prompt in code
    assert "grader.py" not in direct + code
    assert "reference.json" not in direct + code
    assert "absolute_tolerance" not in direct + code
    assert hidden_reference not in direct + code


def test_direct_answer_is_exactly_one_call_without_execution_or_repair(
    task, tmp_path: Path, config: BenchmarkConfig
) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    model = ScriptedRoleModel({"direct_answer": [_final(2.0)]})
    with patch.object(LocalPythonRunner, "run") as execute:
        outcome = run_direct_answer(
            public=public,
            model=model,
            run_directory=tmp_path / "attempt",
            config=config,
        )

    execute.assert_not_called()
    assert len(model.calls) == 1
    assert model.calls[0].role == "direct_answer"
    assert outcome.status == "completed"
    assert outcome.api_call_count == 1
    assert outcome.generated_script_count == 0
    assert outcome.local_repair_count == 0


@pytest.mark.parametrize(
    ("response", "status"),
    [(_final(3.0), "completed"), ("not json", "invalid_json")],
)
def test_direct_answer_wrong_and_invalid_are_not_retried(
    task, tmp_path: Path, config: BenchmarkConfig, response: str, status: str
) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    model = ScriptedRoleModel({"direct_answer": [response]})
    outcome = run_direct_answer(
        public=public,
        model=model,
        run_directory=tmp_path / "attempt",
        config=config,
    )
    assert outcome.status == status
    assert len(model.calls) == 1


def test_direct_answer_context_limit_is_not_applicable_without_call(
    task, tmp_path: Path, config: BenchmarkConfig
) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    model = ScriptedRoleModel({"direct_answer": [_final(2.0)]})
    limited = config.model_copy(update={"direct_answer_max_input_chars": 1})
    outcome = run_direct_answer(
        public=public,
        model=model,
        run_directory=tmp_path / "attempt",
        config=limited,
    )
    assert outcome.status == "not_applicable"
    assert outcome.api_call_count == 0
    assert model.calls == []


def test_one_shot_code_calls_once_executes_once_and_never_verifies_or_repairs(
    task, tmp_path: Path, config: BenchmarkConfig
) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    model = ScriptedRoleModel({"one_shot_code": [f"print({_final(2.0)!r})\n"]})
    outcome = run_one_shot_code(
        public=public,
        model=model,
        run_directory=tmp_path / "attempt",
        config=config,
    )

    scripts = list((tmp_path / "attempt").rglob("generated_code_v*.py"))
    assert outcome.status == "completed"
    assert outcome.generated_script_count == 1
    assert outcome.local_repair_count == 0
    assert len(scripts) == 1
    assert [call.role for call in model.calls] == ["one_shot_code"]


def test_one_shot_failure_is_not_repaired(
    task, tmp_path: Path, config: BenchmarkConfig
) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    model = ScriptedRoleModel({"one_shot_code": ["raise RuntimeError('boom')\n"]})
    outcome = run_one_shot_code(
        public=public,
        model=model,
        run_directory=tmp_path / "attempt",
        config=config,
    )
    assert outcome.status == "execution_failed"
    assert len(model.calls) == 1
    assert outcome.generated_script_count == 1
    assert outcome.local_repair_count == 0


def test_one_shot_invalid_json_is_recorded_without_retry(
    task, tmp_path: Path, config: BenchmarkConfig
) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    model = ScriptedRoleModel({"one_shot_code": ["print('not json')\n"]})
    outcome = run_one_shot_code(
        public=public,
        model=model,
        run_directory=tmp_path / "attempt",
        config=config,
    )
    assert outcome.status == "invalid_json"
    assert len(model.calls) == 1
    assert outcome.generated_script_count == 1


def test_one_shot_prompt_paths_are_usable_for_every_public_input(
    tmp_path: Path, config: BenchmarkConfig
) -> None:
    attempt = tmp_path / "attempt"
    public = stage_public_task(_three_file_public_task(), attempt)
    model = ScriptedRoleModel(
        {"one_shot_code": [_read_all_files_code(attempt, legacy_stdout=True)]}
    )

    outcome = run_one_shot_code(
        public=public,
        model=model,
        run_directory=attempt,
        config=config,
    )

    prompt = model.calls[0].messages[1]["content"]
    assert outcome.status == "completed"
    assert outcome.candidate["key_results"] == {"files_read": 3}
    for path in public.data_files:
        assert path.startswith("inputs/")
        assert path in prompt


def test_staging_and_python_policy_exclude_private_files(task, tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    staged_names = [path.name for path in attempt.rglob("*") if path.is_file()]
    assert "grader.py" not in staged_names
    assert "reference.json" not in staged_names
    assert all("private" not in Path(path).parts for path in public.data_files)

    private_reference = Path(task.private.reference_path)
    result = LocalPythonRunner().run(
        code=(
            f"print(open({str(private_reference)!r}).read())\n"
            "__agent_result__ = {}\n"
        ),
        goal_directory=attempt / "execution",
        allowed_files=[(attempt / path).resolve() for path in public.data_files],
        version=1,
        working_directory=attempt,
    )
    assert not result.success
    assert "not staged" in (result.error or "")


def test_deterministic_grader_success_wrong_missing_invalid_and_tolerance(task) -> None:
    correct = json.loads(_final(2.0))
    wrong = json.loads(_final(2.1))
    missing = {"status": "completed", "answer": "none", "limitations": []}

    assert grade_candidate(correct, task.private).passed
    assert not grade_candidate(wrong, task.private).passed
    assert "numerical mismatch" in grade_candidate(wrong, task.private).errors[0]
    assert not grade_candidate(missing, task.private).passed
    assert not invalid_candidate_grade("invalid JSON").passed
    assert numeric_match(2.0 + 5e-13, 2.0, 1e-12)
    assert not numeric_match(2.0 + 2e-12, 2.0, 1e-12)
    assert compare_values([2, 1], [1, 2], unordered=True)


def _wrong_agent_factory(approach, public):
    assert approach == "agent"
    path = public.data_files[0]
    plan = json.dumps(
        {
            "scientific_objective": "Compute the statistic.",
            "goals": [
                {
                    "goal_id": "calculate",
                    "objective": "Compute the requested statistic.",
                    "required_outputs": ["mean_absolute_successive_difference"],
                    "constraints": [],
                    "success_criteria": ["Report it."],
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
            "concise_reason": "Generate code.",
        }
    )
    code = "__agent_result__ = {'mean_absolute_successive_difference': 3.0}\n"
    generation = json.dumps(
        {
            "kind": "python",
            "code": code,
            "summary": f"Compute from {path}.",
        }
    )
    verifier = '{"decision":"PASS","feedback":"Looks complete."}'
    return ScriptedRoleModel(
        {
            "planner": [plan],
            "executor": [strategy, generation],
            "verifier": [verifier],
        }
    )


def test_agent_uses_full_workflow_but_external_grade_overrides_verifier_pass(
    tmp_path: Path,
) -> None:
    config = BenchmarkConfig(
        model="offline",
        task_ids=["successive_difference_smoke"],
        approaches=["agent"],
    )
    summary, results = run_benchmark(
        config=config,
        output_root=tmp_path / "runs",
        model_factory=_wrong_agent_factory,
        project_root=tmp_path,
    )
    result = results[0]
    assert result.verifier_decisions == ["PASS"]
    assert not result.graded_success
    assert result.status == "wrong_answer"
    assert summary.metrics["agent"].passed_runs == 0


def test_replan_decision_is_not_an_external_grade(task) -> None:
    wrong = json.loads(_final(3.0))
    grade = grade_candidate(wrong, task.private)
    verifier_decisions = ["REPLAN"]
    assert verifier_decisions == ["REPLAN"]
    assert not grade.passed
    assert all("REPLAN" not in error for error in grade.errors)


def test_agent_messages_and_log_never_contain_private_ground_truth(
    task, tmp_path: Path, config: BenchmarkConfig
) -> None:
    public = stage_public_task(task.public, tmp_path / "attempt")
    model = _offline_model_factory("agent", public)
    outcome = run_agent(
        public=public,
        model=model,
        run_directory=tmp_path / "attempt",
        config=config,
    )
    assert outcome.status == "completed"
    messages = json.dumps([call.messages for call in model.calls])
    log = (tmp_path / "attempt/agent_run/workflow.log").read_text(encoding="utf-8")
    for hidden in ("absolute_tolerance", "reference.json", "grader.py"):
        assert hidden not in messages
        assert hidden not in log
    assert {call.role for call in model.calls} == {"planner", "executor", "verifier"}


def test_agent_generated_python_receives_usable_relative_public_paths(
    tmp_path: Path, config: BenchmarkConfig
) -> None:
    attempt = tmp_path / "attempt"
    public = stage_public_task(_three_file_public_task(), attempt)
    plan = json.dumps(
        {
            "scientific_objective": "Read all public inputs.",
            "goals": [
                {
                    "goal_id": "read_inputs",
                    "objective": "Read every supplied public CSV.",
                    "required_outputs": ["files_read"],
                    "constraints": [],
                    "success_criteria": ["All three files are read."],
                    "depends_on": [],
                }
            ],
        }
    )
    strategy = json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": "profile_table",
            "arguments": {"file_path": "inputs/patients.csv"},
            "concise_reason": "Generated code is required.",
        }
    )
    verifier = '{"decision":"PASS","feedback":"All inputs were read."}'
    model = ScriptedRoleModel(
        {
            "planner": [plan],
            "executor": [
                strategy,
                json.dumps(
                    {
                        "kind": "python",
                        "code": _read_all_files_code(attempt),
                        "summary": "Read all staged files.",
                    }
                ),
            ],
            "verifier": [verifier],
        }
    )

    outcome = run_agent(
        public=public,
        model=model,
        run_directory=attempt,
        config=config,
    )

    code_call = [call for call in model.calls if call.role == "executor"][1]
    assert outcome.status == "completed"
    for path in public.data_files:
        assert str((attempt / path).resolve()) in code_call.messages[1]["content"]
    workflow_log = (attempt / "agent_run/workflow.log").read_text(encoding="utf-8")
    assert "Normalized generated_python capability_name" in workflow_log


def test_failed_agent_call_does_not_leave_empty_agent_run(
    task, tmp_path: Path, config: BenchmarkConfig
) -> None:
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    model = ScriptedRoleModel({})

    outcome = run_agent(
        public=public,
        model=model,
        run_directory=attempt,
        config=config,
    )

    assert outcome.status == "error"
    assert not (attempt / "agent_run").exists()


def test_orchestrator_isolates_attempts_persists_rows_and_summarizes_offline(
    tmp_path: Path,
) -> None:
    config = BenchmarkConfig(
        model="offline",
        task_ids=["successive_difference_smoke"],
        approaches=["direct_answer", "one_shot_code", "agent"],
        repeats=2,
    )
    with (
        patch("data_analysis_agent.benchmark.load_settings") as settings,
        patch("data_analysis_agent.benchmark.create_nebius_client") as client,
    ):
        summary, results = run_benchmark(
            config=config,
            output_root=tmp_path / "benchmark_runs",
            project_root=tmp_path,
        )
    settings.assert_not_called()
    client.assert_not_called()
    assert len(results) == 6
    result_path = tmp_path / summary.results_path
    assert len(result_path.read_text(encoding="utf-8").splitlines()) == 6
    assert len({result.artifact_directory for result in results}) == 6
    staged_contents = {
        tuple(
            path.read_text(encoding="utf-8")
            for path in sorted(
                (tmp_path / result.artifact_directory / "inputs").rglob("*")
            )
            if path.is_file()
        )
        for result in results
    }
    smoke_data = (
        DEFAULT_TASKS_ROOT
        / "successive_difference_smoke/public/data/measurements_with_missing.csv"
    ).read_text(encoding="utf-8")
    assert staged_contents == {(smoke_data,)}
    assert all(result.graded_success for result in results)
    assert summary.metrics["direct_answer"].average_api_calls == 1
    assert summary.metrics["one_shot_code"].average_generated_script_versions == 1
    assert summary.metrics["agent"].average_api_calls == 4
    output = format_benchmark_summary(summary, results)
    assert "Complete input data" not in output
    assert "mean_absolute_successive_difference = 2.0" not in output
    assert "direct_answer" in output


def test_benchmark_runs_directory_is_gitignored() -> None:
    gitignore = (DEFAULT_TASKS_ROOT.parent / ".gitignore").read_text(encoding="utf-8")
    assert "benchmark_runs/" in gitignore.splitlines()


class _TransportFailureModel:
    def generate(self, *, role, messages):
        del role, messages
        error_type = type("APIConnectionError", (Exception,), {})
        raise error_type("connection failed")


def test_transport_failure_is_ungraded_infrastructure_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = BenchmarkConfig(
        model="offline",
        task_ids=["successive_difference_smoke"],
        approaches=["direct_answer"],
        live=True,
    )

    summary, results = run_benchmark(
        config=config,
        output_root=tmp_path / "runs",
        model_factory=lambda approach, public: _TransportFailureModel(),
        project_root=tmp_path,
    )

    result = results[0]
    output = capsys.readouterr().out
    assert result.status == "infrastructure_error"
    assert result.error_category == "transport_api"
    assert result.exception_class == "APIConnectionError"
    assert result.api_call_count == 1
    assert result.wall_clock_latency >= 0
    assert not result.graded
    assert result.grader_score is None
    assert result.grader_errors == []
    assert "BENCHMARK RUN 1/1" in output
    assert "Direct answer — calling model" in output
    assert "Grading skipped — infrastructure error" in output
    grade_path = (
        tmp_path / summary.results_path
    ).parent / "direct_answer/successive_difference_smoke/repeat_001/grade.json"
    assert json.loads(grade_path.read_text(encoding="utf-8"))["graded"] is False


def test_policy_failure_has_its_own_status_and_summary_reason(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        model="offline",
        task_ids=["successive_difference_smoke"],
        approaches=["one_shot_code"],
    )
    dynamic_code = (
        "import pandas as pd\n"
        "def read(path):\n"
        "    return pd.read_csv(path)\n"
        "read('inputs/data.csv')\n"
    )

    summary, results = run_benchmark(
        config=config,
        output_root=tmp_path / "runs",
        model_factory=lambda approach, public: ScriptedRoleModel(
            {"one_shot_code": [dynamic_code]}
        ),
        project_root=tmp_path,
    )

    result = results[0]
    rendered = format_benchmark_summary(summary, results)
    assert result.status == "python_policy_failure"
    assert result.error_category == "python_policy"
    assert "python policy failure" in rendered
    assert "PythonPolicyError: Dynamic file paths are prohibited" in rendered


def test_live_progress_reports_successful_model_and_grading_stages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = BenchmarkConfig(
        model="offline",
        task_ids=["successive_difference_smoke"],
        approaches=["direct_answer"],
        live=True,
    )

    run_benchmark(
        config=config,
        output_root=tmp_path / "runs",
        model_factory=lambda approach, public: ScriptedRoleModel(
            {"direct_answer": [_final(2.0)]}
        ),
        project_root=tmp_path,
    )

    output = capsys.readouterr().out
    assert output.count("BENCHMARK RUN 1/1") == 1
    assert "[1/1] direct_answer" not in output
    assert "Direct answer — calling model" in output
    assert "Direct answer — completed" in output
    assert "Grading — starting" in output
    assert "Grading — completed" in output


@pytest.mark.parametrize(
    ("approaches", "scope"),
    [
        (["direct_answer", "one_shot_code", "agent"], "three_way"),
        (["agent"], "agent_only"),
        (["direct_answer", "one_shot_code"], "direct_answer-vs-one_shot_code"),
    ],
)
def test_benchmark_run_id_identifies_task_and_scope(approaches, scope: str) -> None:
    config = BenchmarkConfig(
        model="offline",
        task_ids=["successive_difference_smoke"],
        approaches=approaches,
    )

    run_id = build_benchmark_run_id(config, timestamp="20260715T120000Z")

    assert benchmark_scope_label(config) == scope
    assert run_id == (
        f"benchmark__successive_difference_smoke__{scope}__20260715T120000Z"
    )
