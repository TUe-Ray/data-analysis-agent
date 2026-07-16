"""Build deterministic static and gated adaptive-biomarker benchmark packages."""
# ruff: noqa: E501, E701, E702

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmark_tasks"
STATIC = "adaptive_biomarker_response_static"
GATED = "adaptive_biomarker_response_gated"
STAGE_1 = [
    "protocol/base_protocol.md",
    "protocol/amendment_01.md",
    "documentation/data_dictionary.csv",
    "documentation/value_codebook.csv",
    "release/release_manifest.json",
    "stage_1/specimen_manifest.csv",
    "stage_1/plate_controls.csv",
    "stage_1/consent_and_arm.csv",
]
STAGE_2 = ["stage_2/assay_measurements.csv", "stage_2/assay_run_metadata.csv"]
STAGE_3 = [
    "stage_3/clinical_outcomes.csv",
    "stage_3/source_subject_crosswalk.csv",
    "stage_3/exclusion_events.csv",
]

DATA = {
    "protocol/base_protocol.md": "# Base protocol\nQC >= 0.80 and day-28 outcomes apply unless amended.\n",
    "protocol/amendment_01.md": "# Amendment 01\nWhere this amendment conflicts with the base protocol, it governs. Accept Q_OK and Q_REVIEWED, reject Q_FAIL; QC is >= 0.85. Normalize by mean usable controls. Deduplicate scientific record_id before nonfinite filtering. Select nearest day 35 in day 28 through day 42 inclusive, then quality and record id. Exclude start_date < event_date <= selected visit. Report B minus A.\n",
    "documentation/data_dictionary.csv": "table,column,meaning\nspecimen_manifest,qc_code,physical code\nassay_measurements,record_id,scientific identity\n",
    "documentation/value_codebook.csv": "field,physical_value,semantic_label\nqc_code,Q_OK,valid\nqc_code,Q_REVIEWED,reviewed\nqc_code,Q_FAIL,rejected\n",
    "release/release_manifest.json": '{"rule": "verified artifact columns release public stage"}\n',
    "stage_1/specimen_manifest.csv": "analysis_subject_id,specimen_id,plate_id,qc_code,qc_value\nA01,S01,P1,Q_OK,0.85\nA02,S02,P1,Q_REVIEWED,0.90\nB01,S03,P2,Q_OK,0.86\nB02,S04,P2,Q_FAIL,0.99\n",
    "stage_1/plate_controls.csv": "plate_id,control_id,control_value,usable\nP1,C1,10,yes\nP1,C2,10,yes\nP2,C1,20,yes\nP2,C2,20,no\n",
    "stage_1/consent_and_arm.csv": "analysis_subject_id,arm_code,consented\nA01,A,yes\nA02,A,yes\nB01,B,yes\nB02,B,yes\n",
    "stage_2/assay_measurements.csv": "record_id,specimen_id,measurement\nR1,S01,12\nR1,S01,12\nR2,S02,11\nR3,S03,inf\nR4,S03,22\n",
    "stage_2/assay_run_metadata.csv": "record_id,run_id,technical_duplicate\nR1,RUN1,no\nR2,RUN1,no\nR3,RUN2,no\nR4,RUN2,no\n",
    "stage_3/clinical_outcomes.csv": "source_subject_key,visit_record_id,day,response,quality\nX01,V1,35,5,high\nX02,V2,28,4,low\nX03,V3,42,9,high\n",
    "stage_3/source_subject_crosswalk.csv": "source_subject_key,analysis_subject_id\nX01,A01\nX02,A02\nX03,B01\n",
    "stage_3/exclusion_events.csv": "analysis_subject_id,event_date\nA02,2024-02-01\nB01,2024-03-01\n",
}

SCHEMA = {
    "type": "object",
    "required": ["status", "answer", "key_results", "limitations"],
    "properties": {
        "status": {"enum": ["completed", "completed_with_limitations"]},
        "answer": {"type": "string"},
        "limitations": {"type": "array"},
        "key_results": {
            "type": "object",
            "required": [
                "release_audit",
                "qc_attrition",
                "analysis_attrition",
                "arm_biomarker_summaries",
                "response_summaries",
                "between_arm_comparison",
                "selected_subject_audit_rows",
            ],
        },
    },
}
REFERENCE = {
    "absolute_tolerance": 1e-9,
    "key_results": {
        "release_audit": {"released_stages": ["stage_2", "stage_3"]},
        "qc_attrition": {"input_specimens": 4, "approved_specimens": 2},
        "analysis_attrition": {
            "qc_approved": 2,
            "with_finite_biomarker": 1,
            "with_selected_outcome": 1,
        },
        "arm_biomarker_summaries": {
            "A": {"n": 1, "mean": 1.2},
            "B": {"n": 0, "mean": None},
        },
        "response_summaries": {"A": {"n": 1, "mean": 5.0}, "B": {"n": 0, "mean": None}},
        "between_arm_comparison": {"b_minus_a": None},
        "selected_subject_audit_rows": [
            {
                "analysis_subject_id": "A01",
                "selected_assay_record_id": "R1",
                "selected_visit_record_id": "V1",
            }
        ],
    },
}
GRADER = "from data_analysis_agent.benchmark_types import GradeResult\ndef grade(candidate, reference):\n actual=candidate.get('key_results') if isinstance(candidate,dict) else None\n ok=actual == reference['key_results']\n return GradeResult(passed=ok, score=1.0 if ok else 0.0, errors=[] if ok else ['scientific JSON mismatch'])\n"


def generated_files() -> dict[str, str]:
    prompt = "Reconcile the amendment, then analyze adaptive biomarker response. The gated variant reveals later public data only after verifier-approved QC and normalized-biomarker artifacts. Use the six-stage workflow and return the exact JSON schema.\n"
    output: dict[str, str] = {}
    for task_id in (STATIC, GATED):
        initial = STAGE_1 + STAGE_2 + STAGE_3 if task_id == STATIC else STAGE_1
        config = {
            "public_files": initial,
            "answer_schema": SCHEMA,
            "metadata": {
                "document_files": [
                    "protocol/base_protocol.md",
                    "protocol/amendment_01.md",
                ],
                "document_precedence": ["amendment_01", "base_protocol"],
                "validation_contract": {
                    "required_result_sections": [
                        "release_audit",
                        "qc_attrition",
                        "analysis_attrition",
                    ]
                },
            },
        }
        if task_id == GATED:
            config.update(
                {
                    "initial_public_files": STAGE_1,
                    "deferred_public_files": STAGE_2 + STAGE_3,
                    "release_stages": [
                        {
                            "name": "stage_2",
                            "files": STAGE_2,
                            "required_artifact_columns": [
                                "analysis_subject_id",
                                "specimen_id",
                                "plate_id",
                                "arm_code",
                                "qc_status",
                            ],
                        },
                        {
                            "name": "stage_3",
                            "files": STAGE_3,
                            "required_artifact_columns": [
                                "analysis_subject_id",
                                "normalized_biomarker",
                                "selected_assay_record_id",
                            ],
                        },
                    ],
                }
            )
        output[f"{task_id}/public/task.json"] = json.dumps(config, indent=2) + "\n"
        output[f"{task_id}/public/prompt.txt"] = prompt
        for name in STAGE_1 + STAGE_2 + STAGE_3:
            output[f"{task_id}/public/{name}"] = DATA[name]
        output[f"{task_id}/private/reference.json"] = (
            json.dumps(REFERENCE, indent=2) + "\n"
        )
        output[f"{task_id}/private/grader.py"] = GRADER
    return output


def write_task() -> None:
    for task_id in (STATIC, GATED):
        shutil.rmtree(ROOT / task_id, ignore_errors=True)
    for name, content in generated_files().items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        actual = {
            str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
            for task_id in (STATIC, GATED)
            for path in (ROOT / task_id).rglob("*")
            if path.is_file()
        }
        raise SystemExit(
            0
            if actual == generated_files()
            else "adaptive biomarker task files are stale; run builder"
        )
    write_task()
