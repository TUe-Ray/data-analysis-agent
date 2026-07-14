from __future__ import annotations

from pathlib import Path

import pytest

from data_analysis_agent.evaluation import (
    EvaluationJudgment,
    VerifierCase,
    calculate_metrics,
    format_evaluation_summary,
    load_verifier_cases,
    run_verifier_evaluation,
)
from data_analysis_agent.models import Role

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "examples/verifier_cases.json"


class SequenceVerifierModel:
    """Offline-only fake that returns or raises each configured item in order."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.roles: list[Role] = []

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        del messages
        self.roles.append(role)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_judgment(
    case: VerifierCase, actual: str | None, status: str
) -> EvaluationJudgment:
    return EvaluationJudgment(
        case_id=case.case_id,
        category=case.category,
        repeat_index=1,
        timestamp="2026-01-01T00:00:00+00:00",
        expected_decision=case.expected_decision,
        expected_issue=case.expected_issue,
        actual_decision=actual,
        status=status,
        feedback="concise feedback",
        staged_file_names=["measurements_with_missing.csv"],
        staged_input_context="File: measurements_with_missing.csv\nvalue\n10",
        verifier_messages=[],
        raw_responses=[],
        latency_seconds=0.1,
        error="API error" if status == "ERROR" else None,
    )


def test_case_fixture_loads_and_validates() -> None:
    cases = load_verifier_cases(CASES_PATH)

    assert len(cases) == 10
    assert all(case.question and case.plan and case.execution_result for case in cases)


def test_case_ids_are_unique_and_gold_labels_are_valid() -> None:
    cases = load_verifier_cases(CASES_PATH)
    case_ids = [case.case_id for case in cases]

    assert len(case_ids) == len(set(case_ids))
    assert {case.expected_decision for case in cases} <= {"PASS", "REPLAN"}
    assert sum(case.expected_decision == "PASS" for case in cases) >= 2
    assert sum(case.expected_decision == "REPLAN" for case in cases) >= 2


def test_metric_and_false_rate_calculations() -> None:
    all_cases = load_verifier_cases(CASES_PATH)
    pass_case = all_cases[0]
    replan_case = all_cases[1]
    cases = [pass_case, replan_case]
    judgments = [
        make_judgment(pass_case, "PASS", "CORRECT"),
        make_judgment(pass_case, "REPLAN", "INCORRECT"),
        make_judgment(replan_case, "REPLAN", "CORRECT"),
        make_judgment(replan_case, "PASS", "INCORRECT"),
    ]

    metrics = calculate_metrics(cases, judgments)

    assert metrics.total_judgments == 4
    assert metrics.correct_judgments == 2
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.true_accepts == 1
    assert metrics.true_rejects == 1
    assert metrics.false_accepts == 1
    assert metrics.false_acceptance_rate == pytest.approx(0.5)
    assert metrics.false_rejects == 1
    assert metrics.false_rejection_rate == pytest.approx(0.5)


def test_case_error_does_not_stop_evaluation_and_logs_details(tmp_path: Path) -> None:
    cases = load_verifier_cases(CASES_PATH)[:2]
    model = SequenceVerifierModel(
        [
            RuntimeError("temporary API error"),
            '{"decision":"REPLAN","feedback":"The standard error is missing."}',
        ]
    )

    run = run_verifier_evaluation(
        cases=cases,
        cases_path=CASES_PATH,
        model=model,
        model_id="fake-verifier",
        repeats=1,
        output_dir=tmp_path,
        project_root=PROJECT_ROOT,
    )

    assert [judgment.status for judgment in run.judgments] == ["ERROR", "CORRECT"]
    assert run.metrics.errors == 1
    assert run.metrics.evaluated_judgments == 1
    assert model.roles == ["verifier", "verifier"]

    detailed_log = run.log_path.read_text(encoding="utf-8")
    for required_text in (
        "Question:",
        "Fixed plan:",
        "Execution result:",
        "Expected decision:",
        "Raw model responses:",
        "Parsed Pydantic result:",
    ):
        assert required_text in detailed_log


def test_terminal_formatter_is_readable_and_hides_detailed_inputs(
    tmp_path: Path,
) -> None:
    cases = load_verifier_cases(CASES_PATH)[:1]
    model = SequenceVerifierModel(
        ['{"decision":"PASS","feedback":"All requested values are present."}']
    )
    run = run_verifier_evaluation(
        cases=cases,
        cases_path=CASES_PATH,
        model=model,
        model_id="fake-verifier",
        repeats=1,
        output_dir=tmp_path,
        project_root=PROJECT_ROOT,
    )

    output = format_evaluation_summary(run)

    assert "VERIFIER INTELLIGENCE EVALUATION" in output
    assert "CASE 01" in output
    assert "SUMMARY" in output
    assert "=" * 60 in output
    assert "Raw model responses" not in output
    assert "You are the independent scientific Verifier" not in output
    assert "NEBIUS_API_KEY" not in output
