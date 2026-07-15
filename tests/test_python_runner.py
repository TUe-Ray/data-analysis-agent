import json
from pathlib import Path
from unittest.mock import patch

import pytest

from data_analysis_agent.python_runner import LocalPythonRunner, validate_generated_code


def test_runner_captures_success_and_saves_artifacts(tmp_path: Path) -> None:
    result = LocalPythonRunner(timeout_seconds=2).run(
        code='print("debug")\n__agent_result__ = {"result": 2.0}\n',
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
    assert (tmp_path / "goal/generated_outputs/result.json").read_text(
        encoding="utf-8"
    ) == '{"result":2.0}'


def test_runner_handles_timeout(tmp_path: Path) -> None:
    result = LocalPythonRunner(timeout_seconds=0.05).run(
        code="import time\ntime.sleep(1)\n",
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
        result_mode="legacy_stdout",
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
        result_mode="legacy_stdout",
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
        result_mode="legacy_stdout",
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
        result_mode="legacy_stdout",
    )
    second = runner.run(
        code="__agent_result__ = {}\n",
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
        result_mode="legacy_stdout",
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
        result_mode="legacy_stdout",
    )

    assert result.success
    assert (goal / "generated_outputs/result.json").is_file()


def test_runner_accepts_exact_staged_absolute_pandas_path(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    staged = attempt / "inputs/patients.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("patient_id\nP001\n", encoding="utf-8")

    result = LocalPythonRunner().run(
        code=(
            "import pandas as pd\n"
            f"frame = pd.read_csv({str(staged.resolve())!r})\n"
            "__agent_result__ = {'rows': len(frame)}\n"
        ),
        goal_directory=attempt / "execution",
        allowed_files=[staged.resolve()],
        version=1,
    )

    assert result.success
    assert result.result == {"rows": 1}


@pytest.mark.parametrize(
    "path_expression_kind",
    [
        "literal",
        "name",
        "path_literal",
        "path_join",
        "mapping",
    ],
)
def test_runner_accepts_statically_resolvable_staged_paths(
    tmp_path: Path, path_expression_kind: str
) -> None:
    attempt = tmp_path / "attempt"
    staged = attempt / "inputs/patients.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("patient_id\nP001\n", encoding="utf-8")
    path_expression = {
        "literal": repr(str(staged.resolve())),
        "name": "PATIENTS_PATH",
        "path_literal": f"Path({str(staged.resolve())!r})",
        "path_join": "INPUTS / 'patients.csv'",
        "mapping": "PATHS['patients']",
    }[path_expression_kind]
    code = (
        "import pandas as pd\n"
        "from pathlib import Path\n"
        f"PATIENTS_PATH = {str(staged.resolve())!r}\n"
        f"INPUTS = Path({str(staged.parent.resolve())!r})\n"
        f"PATHS = {{'patients': {str(staged.resolve())!r}}}\n"
        f"frame = pd.read_csv({path_expression})\n"
        "__agent_result__ = {'rows': len(frame)}\n"
    )

    result = LocalPythonRunner().run(
        code=code,
        goal_directory=attempt / "execution",
        allowed_files=[staged.resolve()],
        version=1,
    )

    assert result.success
    assert result.result == {"rows": 1}


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (
            "import pandas as pd\n"
            "for path in ['inputs/patients.csv']:\n"
            "    pd.read_csv(path)\n"
            "print('{}')\n",
            "Dynamic file paths",
        ),
        (
            "import os\nimport pandas as pd\n"
            "pd.read_csv(os.environ['DATA_PATH'])\nprint('{}')\n",
            "Environment-variable access",
        ),
        (
            "import glob\nimport pandas as pd\n"
            "pd.read_csv(glob.glob('inputs/*.csv')[0])\nprint('{}')\n",
            "Path discovery",
        ),
    ],
)
def test_runner_rejects_dynamic_or_discovered_read_paths(
    tmp_path: Path, code: str, reason: str
) -> None:
    attempt = tmp_path / "attempt"
    staged = attempt / "inputs/patients.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("patient_id\nP001\n", encoding="utf-8")

    result = LocalPythonRunner().run(
        code=code,
        goal_directory=attempt / "execution",
        working_directory=attempt,
        allowed_files=[staged.resolve()],
        version=1,
        result_mode="legacy_stdout",
    )

    assert not result.success
    assert reason in (result.error or "")


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
        result_mode="legacy_stdout",
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
        result_mode="legacy_stdout",
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
        result_mode="legacy_stdout",
    )

    assert not result.success
    assert "Prohibited file operation: replace" in (result.error or "")


@pytest.mark.parametrize(
    "code",
    [
        (
            "import pandas as pd\n"
            "def load_csv(path):\n"
            "    return pd.read_csv(path)\n"
            "load_csv('inputs/patients.csv')\n"
        ),
        (
            "from pathlib import Path\nimport pandas as pd\n"
            "base = Path(__file__).parent\n"
            "pd.read_csv(base / 'inputs' / 'patients.csv')\n"
        ),
        (
            "import os\nimport pandas as pd\n"
            "path = os.path.join('inputs', 'patients.csv')\n"
            "pd.read_csv(path)\n"
        ),
    ],
)
def test_previous_dynamic_generated_path_patterns_remain_blocked(
    tmp_path: Path, code: str
) -> None:
    attempt = tmp_path / "attempt"
    staged = attempt / "inputs/patients.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("patient_id\nP001\n", encoding="utf-8")

    result = LocalPythonRunner().run(
        code=code,
        goal_directory=attempt / "execution",
        working_directory=attempt,
        allowed_files=[staged.resolve()],
        version=1,
        result_mode="legacy_stdout",
    )

    assert not result.success
    assert "Dynamic file paths" in (result.error or "")


def test_debug_and_multiline_stdout_are_not_authoritative(tmp_path: Path) -> None:
    code = (
        "import json\n"
        "print('debug before')\n"
        "print(json.dumps({'wrong': True}, indent=2))\n"
        "__agent_result__ = {'right': 7}\n"
        "print('debug after')\n"
    )

    result = LocalPythonRunner().run(
        code=code,
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert result.success
    assert result.result == {"right": 7}
    assert "debug before" in result.stdout
    assert "debug after" in result.stdout


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("value = 1\n", "missing executable module-level assignment"),
        ("__agent_result__ = [1, 2]\n", "result is not an object"),
        ("__agent_result__ = {'bad': {1, 2}}\n", "not JSON-serializable"),
        ("__agent_result__ = {'bad': float('nan')}\n", "NaN or Infinity"),
        ("__agent_result__ = {'bad': float('inf')}\n", "NaN or Infinity"),
    ],
)
def test_agent_result_contract_failures_are_typed(
    tmp_path: Path, code: str, message: str
) -> None:
    result = LocalPythonRunner().run(
        code=code,
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert not result.success
    expected_category = (
        "generation_contract_error"
        if "missing executable module-level assignment" in message
        else "result_contract_error"
    )
    assert result.failure_category == expected_category
    assert message in (result.error or "")


def test_trusted_result_file_missing_is_clear(tmp_path: Path) -> None:
    with patch(
        "data_analysis_agent.python_runner._trusted_runner_source",
        return_value="pass\n",
    ):
        result = LocalPythonRunner().run(
            code="__agent_result__ = {'value': 1}\n",
            goal_directory=tmp_path / "goal",
            allowed_files=[],
            version=1,
        )

    assert not result.success
    assert result.failure_category == "result_contract_error"
    assert "trusted result file missing" in (result.error or "")


def test_relative_outputs_are_created_inside_goal_directory(tmp_path: Path) -> None:
    goal = tmp_path / "goal"
    result = LocalPythonRunner().run(
        code=(
            "open('patients_normalized.csv', 'w', encoding='utf-8').write('ok')\n"
            "__agent_result__ = {'saved': True}\n"
        ),
        goal_directory=goal,
        allowed_files=[],
        version=1,
    )

    assert result.success
    assert (goal / "patients_normalized.csv").read_text(encoding="utf-8") == "ok"
    assert not (tmp_path / "patients_normalized.csv").exists()


def test_trusted_result_serialization_is_deterministic(tmp_path: Path) -> None:
    goal = tmp_path / "goal"
    result = LocalPythonRunner().run(
        code="__agent_result__ = {'z': 1, 'a': [2, 3]}\n",
        goal_directory=goal,
        allowed_files=[],
        version=1,
    )

    assert result.success
    raw = (goal / "generated_outputs/result.json").read_text(encoding="utf-8")
    assert raw == '{"a":[2,3],"z":1}'
    assert json.loads(raw) == result.result
