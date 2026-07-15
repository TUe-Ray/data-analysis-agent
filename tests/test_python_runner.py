from pathlib import Path

import pytest

from data_analysis_agent.python_runner import LocalPythonRunner, validate_generated_code


def test_runner_captures_success_and_saves_artifacts(tmp_path: Path) -> None:
    result = LocalPythonRunner(timeout_seconds=2).run(
        code='import json\nprint(json.dumps({"result": 2.0}))\n',
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert result.success
    assert result.exit_code == 0
    assert result.result == {"result": 2.0}
    assert result.duration_seconds >= 0
    assert Path(result.script_path).is_file()
    assert (tmp_path / "goal/execution_result.json").is_file()
    assert (tmp_path / "goal/stdout.txt").read_text(encoding="utf-8")
    assert not (tmp_path / "goal/generated_outputs").exists()


def test_runner_handles_timeout(tmp_path: Path) -> None:
    result = LocalPythonRunner(timeout_seconds=0.05).run(
        code="import time\ntime.sleep(1)\n",
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert not result.success
    assert result.timed_out
    assert "timed out" in (result.error or "")


def test_runner_rejects_prohibited_network_import(tmp_path: Path) -> None:
    result = LocalPythonRunner().run(
        code="import socket\nprint('{}')\n",
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert not result.success
    assert result.exit_code is None
    assert "Prohibited import" in (result.error or "")


def test_runner_rejects_literal_write_outside_run_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    code = f"open({str(outside)!r}, 'w').write('no')\nprint('{{}}')\n"

    result = LocalPythonRunner().run(
        code=code,
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert not result.success
    assert "Writing outside" in (result.error or "")
    assert not outside.exists()


def test_runner_preserves_failed_and_repaired_versions(tmp_path: Path) -> None:
    runner = LocalPythonRunner()
    goal = tmp_path / "goal"
    first = runner.run(
        code="raise RuntimeError('mechanical')\n",
        goal_directory=goal,
        allowed_files=[],
        version=1,
    )
    second = runner.run(
        code="print('{}')\n",
        goal_directory=goal,
        allowed_files=[],
        version=2,
    )

    assert not first.success
    assert second.success
    assert (goal / "generated_code_v1.py").is_file()
    assert (goal / "generated_code_v2.py").is_file()
    assert (goal / "execution_result_v1.json").is_file()
    assert (goal / "execution_result_v2.json").is_file()


def test_runner_rejects_dynamic_read_paths(tmp_path: Path) -> None:
    result = LocalPythonRunner().run(
        code="name = input()\nprint(open(name).read())\n",
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert not result.success
    assert "Dynamic file paths" in (result.error or "")
    assert not (tmp_path / "goal/generated_outputs").exists()


def test_runner_preserves_nonempty_generated_outputs(tmp_path: Path) -> None:
    goal = tmp_path / "goal"
    result = LocalPythonRunner().run(
        code=(
            "from pathlib import Path\n"
            "import json\n"
            "Path('generated_outputs').mkdir()\n"
            "Path('generated_outputs/result.json').write_text("
            "json.dumps({'result': 2.0}), encoding='utf-8')\n"
        ),
        goal_directory=goal,
        allowed_files=[],
        version=1,
    )

    assert result.success
    assert (goal / "generated_outputs/result.json").is_file()


def test_runner_accepts_exact_staged_relative_pandas_path(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    staged = attempt / "inputs/patients.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("patient_id\nP001\n", encoding="utf-8")

    result = LocalPythonRunner().run(
        code=(
            "import json\n"
            "import pandas as pd\n"
            "frame = pd.read_csv('inputs/patients.csv')\n"
            "print(json.dumps({'rows': len(frame)}))\n"
        ),
        goal_directory=attempt / "execution",
        working_directory=attempt,
        allowed_files=[staged.resolve()],
        version=1,
    )

    assert result.success
    assert result.result == {"rows": 1}


def test_runner_rejects_unstaged_basename_when_only_inputs_path_is_staged(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    staged = attempt / "inputs/patients.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("patient_id\nP001\n", encoding="utf-8")

    result = LocalPythonRunner().run(
        code=(
            "import pandas as pd\nframe = pd.read_csv('patients.csv')\nprint('{}')\n"
        ),
        goal_directory=attempt / "execution",
        working_directory=attempt,
        allowed_files=[staged.resolve()],
        version=1,
    )

    assert not result.success
    assert "not staged" in (result.error or "")


@pytest.mark.parametrize("private_name", ["reference.json", "grader.py"])
def test_runner_rejects_private_task_files(tmp_path: Path, private_name: str) -> None:
    attempt = tmp_path / "attempt"
    staged = attempt / "inputs/patients.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("patient_id\nP001\n", encoding="utf-8")
    private = attempt / "private" / private_name
    private.parent.mkdir()
    private.write_text("hidden", encoding="utf-8")

    result = LocalPythonRunner().run(
        code=f"print(open('private/{private_name}').read())\n",
        goal_directory=attempt / "execution",
        working_directory=attempt,
        allowed_files=[staged.resolve()],
        version=1,
    )

    assert not result.success
    assert "not staged" in (result.error or "")


def test_policy_allows_harmless_string_replace(tmp_path: Path) -> None:
    validate_generated_code(
        "value = 'treatment_a'.replace('_', ' ')\nprint('{}')\n",
        run_directory=tmp_path / "execution",
        allowed_files=[],
    )


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.replace('source', 'target')\n",
        "from pathlib import Path\nPath('source').replace('target')\n",
    ],
)
def test_policy_still_rejects_filesystem_replace(tmp_path: Path, code: str) -> None:
    result = LocalPythonRunner().run(
        code=code,
        goal_directory=tmp_path / "execution",
        allowed_files=[],
        version=1,
    )

    assert not result.success
    assert "Prohibited file operation: replace" in (result.error or "")
