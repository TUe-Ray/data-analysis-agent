"""End-to-end deterministic smoke coverage for longitudinal benchmark Goal 1."""

import json
from pathlib import Path

from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.python_runner import LocalPythonRunner


def test_longitudinal_g1_reaches_verifier_pass_with_real_runner(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    public = project_root / "benchmark_tasks/longitudinal_treatment_response/public"
    paths = {
        name: (public / "data" / name).resolve()
        for name in ("patients.csv", "visits.csv", "exclusions.csv")
    }
    objective = (
        "Load the three CSV files and normalize treatment arm labels according "
        "to the mapping rules."
    )
    plan = json.dumps(
        {
            "scientific_objective": objective,
            "goals": [
                {
                    "goal_id": "G1",
                    "objective": objective,
                    "required_outputs": [
                        "normalized treatment arm labels",
                        "input row counts",
                    ],
                    "constraints": [
                        "Use all three supplied CSV files.",
                        "Unknown or missing arms are ineligible.",
                    ],
                    "success_criteria": [
                        "All files are loaded and arm labels map to A, B, or null."
                    ],
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
            "concise_reason": "Normalization across three files needs local code.",
        }
    )
    code = f'''import csv


def normalize_arm(value):
    normalized = (value or "").strip().lower().replace("_", " ")
    if normalized in {{"a", "arm a", "treatment a"}}:
        return "A"
    if normalized in {{"b", "arm b", "treatment b"}}:
        return "B"
    return None


with open({str(paths["patients.csv"])!r}, encoding="utf-8") as handle:
    patients = list(csv.DictReader(handle))
with open({str(paths["visits.csv"])!r}, encoding="utf-8") as handle:
    visits = list(csv.DictReader(handle))
with open({str(paths["exclusions.csv"])!r}, encoding="utf-8") as handle:
    exclusions = list(csv.DictReader(handle))

normalized = [normalize_arm(row.get("treatment_arm")) for row in patients]
with open("patients_normalized.csv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["patient_id", "normalized_arm"])
    writer.writerows(
        [row["patient_id"], arm] for row, arm in zip(patients, normalized)
    )

__agent_result__ = {{
    "patient_rows": len(patients),
    "visit_rows": len(visits),
    "exclusion_rows": len(exclusions),
    "normalized_arm_counts": {{
        "A": normalized.count("A"),
        "B": normalized.count("B"),
        "ineligible": normalized.count(None),
    }},
}}
'''
    generation = json.dumps(
        {
            "kind": "python",
            "code": code,
            "summary": "Load all inputs and normalize treatment arms.",
        }
    )
    verifier = json.dumps(
        {
            "decision": "PASS",
            "feedback": "All three files were loaded and arm labels were normalized.",
        }
    )
    model = ScriptedRoleModel(
        {
            "planner": [plan],
            "executor": [strategy, generation],
            "verifier": [verifier],
        }
    )
    run_directory = tmp_path / "agent_run"

    result = build_graph(model, runner=LocalPythonRunner(timeout_seconds=10)).invoke(
        {
            "question": (public / "prompt.txt").read_text(encoding="utf-8"),
            "file_paths": list(paths),
            "staged_file_paths": [str(path) for path in paths.values()],
            "input_context": "Three staged longitudinal public CSV files.",
            "run_directory": str(run_directory),
            "replan_count": 0,
            "max_replans": 1,
            "max_code_repair_attempts": 6,
            "max_code_repair_no_progress_attempts": 3,
            "trace": [],
        }
    )

    goal_directory = run_directory / "goals/G1"
    assert result["status"] == "completed"
    assert result["generated_execution_history"][-1]["success"] is True
    assert result["verification_decision"] == "PASS"
    assert [item["goal_id"] for item in result["completed_goal_results"]] == ["G1"]
    assert result["replan_count"] == 0
    assert result.get("code_repair_count", 0) == 0
    assert result["code_repair_attempts_for_current_goal"] == 0
    assert result["execution_failure_category"] is None
    assert (goal_directory / "patients_normalized.csv").is_file()
    assert (goal_directory / "generated_outputs/result.json").is_file()
    assert "write" not in (result.get("policy_failure_reason") or "").lower()
