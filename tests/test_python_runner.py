from pathlib import Path

from data_analysis_agent.python_runner import LocalPythonRunner


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
