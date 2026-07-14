"""Small diagnostic evaluation of live Verifier judgment quality."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from data_analysis_agent.models import RoleModel
from data_analysis_agent.prompts import VERIFIER_REPAIR_PROMPT, build_verifier_messages
from data_analysis_agent.schemas import VerificationOutput

Decision = Literal["PASS", "REPLAN"]
JudgmentStatus = Literal["CORRECT", "INCORRECT", "ERROR"]
DIVIDER = "=" * 60
SUBDIVIDER = "-" * 60
MAX_EVALUATION_FILE_BYTES = 50 * 1024


class EvaluationFixtureError(ValueError):
    """Raised when verifier evaluation cases cannot be loaded safely."""


class EvaluationOutputError(OSError):
    """Raised when an evaluation run directory cannot be created or written."""


class VerifierCase(BaseModel):
    """One human-labeled fixed-input Verifier judgment case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    file_paths: list[str] = Field(min_length=1)
    plan: str = Field(min_length=1)
    execution_result: str = Field(min_length=1)
    expected_decision: Decision
    expected_issue: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    notes: str = Field(min_length=1)


class EvaluationJudgment(BaseModel):
    """One judgment, including details retained only in run logs."""

    case_id: str
    category: str
    repeat_index: int
    timestamp: str
    expected_decision: Decision
    expected_issue: str
    actual_decision: Decision | None = None
    status: JudgmentStatus
    feedback: str
    staged_file_names: list[str]
    staged_input_context: str
    verifier_messages: list[list[dict[str, str]]]
    raw_responses: list[str]
    parsed_result: dict[str, str] | None = None
    latency_seconds: float
    token_usage: dict[str, int] | None = None
    error: str | None = None


class EvaluationMetrics(BaseModel):
    """Aggregate diagnostic metrics over successful and failed judgments."""

    total_cases: int
    total_judgments: int
    evaluated_judgments: int
    correct_judgments: int
    accuracy: float
    gold_pass_cases: int
    gold_replan_cases: int
    true_accepts: int
    true_rejects: int
    false_accepts: int
    false_rejects: int
    false_acceptance_rate: float
    false_rejection_rate: float
    errors: int


class CaseAgreement(BaseModel):
    """Repeat-level agreement summary for one fixed case."""

    case_id: str
    judgments: int
    pass_count: int
    replan_count: int
    error_count: int
    agreement_rate: float


@dataclass(frozen=True)
class EvaluationRun:
    """Completed evaluation and its generated diagnostic artifacts."""

    model_id: str
    repeats: int
    cases: list[VerifierCase]
    judgments: list[EvaluationJudgment]
    metrics: EvaluationMetrics
    agreements: list[CaseAgreement]
    run_directory: Path
    log_path: Path
    results_path: Path
    config_path: Path


def load_verifier_cases(path: Path) -> list[VerifierCase]:
    """Load and validate the machine-readable gold case fixture."""
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
        cases = TypeAdapter(list[VerifierCase]).validate_python(raw_cases)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise EvaluationFixtureError(
            f"Could not load verifier case fixture {path}: {error}"
        ) from error
    if not cases:
        raise EvaluationFixtureError("Verifier case fixture must not be empty")
    case_ids = [case.case_id for case in cases]
    duplicates = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    )
    if duplicates:
        raise EvaluationFixtureError(
            f"Verifier case IDs must be unique: {', '.join(duplicates)}"
        )
    return cases


def _stage_case_files(case: VerifierCase, project_root: Path) -> tuple[list[str], str]:
    names: list[str] = []
    sections: list[str] = []
    for relative_path in case.file_paths:
        path = project_root / relative_path
        if not path.is_file():
            raise EvaluationFixtureError(
                f"Case {case.case_id} references a missing file: {relative_path}"
            )
        if path.stat().st_size > MAX_EVALUATION_FILE_BYTES:
            raise EvaluationFixtureError(
                f"Case {case.case_id} file exceeds 50 KB: {relative_path}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise EvaluationFixtureError(
                f"Could not stage {relative_path} for {case.case_id}: {error}"
            ) from error
        names.append(path.name)
        sections.append(f"File: {path.name}\n{content.rstrip()}")
    return names, "\n\n".join(sections)


def _judge_once(
    *,
    case: VerifierCase,
    repeat_index: int,
    input_context: str,
    staged_names: list[str],
    model: RoleModel,
) -> EvaluationJudgment:
    messages = build_verifier_messages(
        question=case.question,
        input_context=input_context,
        plan=case.plan,
        execution_result=case.execution_result,
    )
    sent_messages: list[list[dict[str, str]]] = []
    raw_responses: list[str] = []
    started = time.perf_counter()
    timestamp = datetime.now(UTC).isoformat()
    last_error: str | None = None
    token_usage: Counter[str] = Counter()

    for attempt in range(2):
        sent_messages.append([dict(message) for message in messages])
        try:
            raw_response = model.generate(role="verifier", messages=messages)
        except Exception as error:  # one failed case must not stop the run
            last_error = f"{type(error).__name__}: {error}"
            break
        latest_usage = getattr(model, "last_token_usage", None)
        if isinstance(latest_usage, dict):
            token_usage.update(latest_usage)
        raw_responses.append(raw_response)
        try:
            parsed = VerificationOutput.model_validate_json(raw_response)
        except ValidationError as error:
            last_error = f"ValidationError: {error}"
            if attempt == 0:
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw_response},
                    {"role": "user", "content": VERIFIER_REPAIR_PROMPT},
                ]
            continue

        status: JudgmentStatus = (
            "CORRECT" if parsed.decision == case.expected_decision else "INCORRECT"
        )
        return EvaluationJudgment(
            case_id=case.case_id,
            category=case.category,
            repeat_index=repeat_index,
            timestamp=timestamp,
            expected_decision=case.expected_decision,
            expected_issue=case.expected_issue,
            actual_decision=parsed.decision,
            status=status,
            feedback=parsed.feedback,
            staged_file_names=staged_names,
            staged_input_context=input_context,
            verifier_messages=sent_messages,
            raw_responses=raw_responses,
            parsed_result=parsed.model_dump(),
            latency_seconds=time.perf_counter() - started,
            token_usage=dict(token_usage) or None,
        )

    return EvaluationJudgment(
        case_id=case.case_id,
        category=case.category,
        repeat_index=repeat_index,
        timestamp=timestamp,
        expected_decision=case.expected_decision,
        expected_issue=case.expected_issue,
        status="ERROR",
        feedback="Verifier judgment could not be evaluated.",
        staged_file_names=staged_names,
        staged_input_context=input_context,
        verifier_messages=sent_messages,
        raw_responses=raw_responses,
        latency_seconds=time.perf_counter() - started,
        token_usage=dict(token_usage) or None,
        error=last_error or "Unknown verifier evaluation error",
    )


def calculate_metrics(
    cases: list[VerifierCase], judgments: list[EvaluationJudgment]
) -> EvaluationMetrics:
    """Calculate accuracy and false acceptance/rejection diagnostics."""
    evaluated = [judgment for judgment in judgments if judgment.status != "ERROR"]
    true_accepts = sum(
        judgment.expected_decision == judgment.actual_decision == "PASS"
        for judgment in evaluated
    )
    true_rejects = sum(
        judgment.expected_decision == judgment.actual_decision == "REPLAN"
        for judgment in evaluated
    )
    false_accepts = sum(
        judgment.expected_decision == "REPLAN" and judgment.actual_decision == "PASS"
        for judgment in evaluated
    )
    false_rejects = sum(
        judgment.expected_decision == "PASS" and judgment.actual_decision == "REPLAN"
        for judgment in evaluated
    )
    correct = true_accepts + true_rejects
    replan_classifications = true_rejects + false_accepts
    pass_classifications = true_accepts + false_rejects
    return EvaluationMetrics(
        total_cases=len(cases),
        total_judgments=len(judgments),
        evaluated_judgments=len(evaluated),
        correct_judgments=correct,
        accuracy=correct / len(evaluated) if evaluated else 0.0,
        gold_pass_cases=sum(case.expected_decision == "PASS" for case in cases),
        gold_replan_cases=sum(case.expected_decision == "REPLAN" for case in cases),
        true_accepts=true_accepts,
        true_rejects=true_rejects,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        false_acceptance_rate=(
            false_accepts / replan_classifications if replan_classifications else 0.0
        ),
        false_rejection_rate=(
            false_rejects / pass_classifications if pass_classifications else 0.0
        ),
        errors=len(judgments) - len(evaluated),
    )


def calculate_case_agreements(
    cases: list[VerifierCase], judgments: list[EvaluationJudgment]
) -> list[CaseAgreement]:
    """Summarize repeat-level decisions and agreement with each gold label."""
    by_case: dict[str, list[EvaluationJudgment]] = defaultdict(list)
    for judgment in judgments:
        by_case[judgment.case_id].append(judgment)

    agreements: list[CaseAgreement] = []
    for case in cases:
        case_judgments = by_case[case.case_id]
        evaluated = [item for item in case_judgments if item.status != "ERROR"]
        correct = sum(item.status == "CORRECT" for item in evaluated)
        agreements.append(
            CaseAgreement(
                case_id=case.case_id,
                judgments=len(case_judgments),
                pass_count=sum(item.actual_decision == "PASS" for item in evaluated),
                replan_count=sum(
                    item.actual_decision == "REPLAN" for item in evaluated
                ),
                error_count=len(case_judgments) - len(evaluated),
                agreement_rate=correct / len(evaluated) if evaluated else 0.0,
            )
        )
    return agreements


def _create_run_directory(output_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = output_dir / f"verifier_eval_{timestamp}"
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise EvaluationOutputError(
            f"Could not create evaluation output directory {run_directory}: {error}"
        ) from error
    return run_directory


def _write_evaluation_artifacts(run: EvaluationRun, cases_path: Path) -> None:
    cases_by_id = {case.case_id: case for case in run.cases}
    log_lines = [
        "VERIFIER INTELLIGENCE EVALUATION — DETAILED LOG",
        f"Model ID: {run.model_id}",
        f"Repeats: {run.repeats}",
        f"Case fixture: {cases_path}",
        "",
    ]
    for judgment in run.judgments:
        case = cases_by_id[judgment.case_id]
        log_lines.extend(
            [
                SUBDIVIDER,
                f"Case ID: {case.case_id}",
                f"Repeat index: {judgment.repeat_index}",
                f"Timestamp: {judgment.timestamp}",
                f"Category: {case.category}",
                f"Severity: {case.severity}",
                f"Question:\n{case.question}",
                f"Staged file names: {', '.join(judgment.staged_file_names)}",
                f"Staged file content:\n{judgment.staged_input_context}",
                f"Fixed plan:\n{case.plan}",
                f"Execution result:\n{case.execution_result}",
                f"Expected decision: {case.expected_decision}",
                f"Expected issue: {case.expected_issue}",
                "Exact Verifier messages sent:",
                json.dumps(judgment.verifier_messages, indent=2, ensure_ascii=False),
                "Raw model responses:",
                json.dumps(judgment.raw_responses, indent=2, ensure_ascii=False),
                "Parsed Pydantic result:",
                json.dumps(judgment.parsed_result, indent=2, ensure_ascii=False),
                f"Matched gold label: {judgment.status == 'CORRECT'}",
                f"Status: {judgment.status}",
                f"Latency seconds: {judgment.latency_seconds:.6f}",
                f"Token usage: {judgment.token_usage or 'unavailable'}",
                f"Error: {judgment.error or 'none'}",
                "",
            ]
        )

    results_payload = {
        "model_id": run.model_id,
        "repeats": run.repeats,
        "metrics": run.metrics.model_dump(mode="json"),
        "per_case_agreement": [
            agreement.model_dump(mode="json") for agreement in run.agreements
        ],
        "judgments": [judgment.model_dump(mode="json") for judgment in run.judgments],
    }
    config_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": run.model_id,
        "repeats": run.repeats,
        "case_fixture": str(cases_path),
        "case_ids": [case.case_id for case in run.cases],
        "temperature": 0,
    }
    try:
        run.log_path.write_text("\n".join(log_lines), encoding="utf-8")
        run.results_path.write_text(
            json.dumps(results_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        run.config_path.write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise EvaluationOutputError(
            f"Could not write evaluation artifacts in {run.run_directory}: {error}"
        ) from error


def run_verifier_evaluation(
    *,
    cases: list[VerifierCase],
    cases_path: Path,
    model: RoleModel,
    model_id: str,
    repeats: int,
    output_dir: Path,
    project_root: Path,
) -> EvaluationRun:
    """Run fixed cases through only the Verifier and persist artifacts."""
    if repeats < 1:
        raise EvaluationFixtureError("repeats must be at least 1")
    run_directory = _create_run_directory(output_dir)
    judgments: list[EvaluationJudgment] = []
    for case in cases:
        try:
            staged_names, input_context = _stage_case_files(case, project_root)
        except EvaluationFixtureError as error:
            for repeat_index in range(1, repeats + 1):
                judgments.append(
                    EvaluationJudgment(
                        case_id=case.case_id,
                        category=case.category,
                        repeat_index=repeat_index,
                        timestamp=datetime.now(UTC).isoformat(),
                        expected_decision=case.expected_decision,
                        expected_issue=case.expected_issue,
                        status="ERROR",
                        feedback="Case input could not be staged.",
                        staged_file_names=[],
                        staged_input_context="",
                        verifier_messages=[],
                        raw_responses=[],
                        latency_seconds=0.0,
                        error=str(error),
                    )
                )
            continue
        for repeat_index in range(1, repeats + 1):
            judgments.append(
                _judge_once(
                    case=case,
                    repeat_index=repeat_index,
                    input_context=input_context,
                    staged_names=staged_names,
                    model=model,
                )
            )

    metrics = calculate_metrics(cases, judgments)
    run = EvaluationRun(
        model_id=model_id,
        repeats=repeats,
        cases=cases,
        judgments=judgments,
        metrics=metrics,
        agreements=calculate_case_agreements(cases, judgments),
        run_directory=run_directory,
        log_path=run_directory / "evaluation.log",
        results_path=run_directory / "results.json",
        config_path=run_directory / "run_config.json",
    )
    _write_evaluation_artifacts(run, cases_path)
    return run


def format_evaluation_summary(run: EvaluationRun) -> str:
    """Format a concise terminal report without prompts, raw output, or secrets."""
    input_names = sorted(
        {name for judgment in run.judgments for name in judgment.staged_file_names}
    )
    lines = [
        DIVIDER,
        "VERIFIER INTELLIGENCE EVALUATION",
        DIVIDER,
        f"Model: {run.model_id}",
        f"Cases: {len(run.cases)}",
        f"Repeats: {run.repeats}",
        f"Input data: {', '.join(input_names) or 'unavailable'}",
    ]
    for index, judgment in enumerate(run.judgments, start=1):
        lines.extend(
            [
                "",
                SUBDIVIDER,
                (
                    f"CASE {index:02d} — {judgment.case_id} "
                    f"(repeat {judgment.repeat_index})"
                ),
                SUBDIVIDER,
                f"Category : {judgment.category}",
                f"Expected : {judgment.expected_decision}",
                f"Actual   : {judgment.actual_decision or 'ERROR'}",
                f"Result   : {judgment.status}",
                f"Feedback : {judgment.feedback.replace(chr(10), ' ')}",
            ]
        )
    metrics = run.metrics
    lines.extend(
        [
            "",
            DIVIDER,
            "SUMMARY",
            DIVIDER,
            f"Total judgments       : {metrics.total_judgments}",
            f"Evaluated judgments   : {metrics.evaluated_judgments}",
            f"Correct judgments     : {metrics.correct_judgments}",
            f"Accuracy              : {metrics.accuracy:.1%}",
            f"Gold PASS cases       : {metrics.gold_pass_cases}",
            f"Gold REPLAN cases     : {metrics.gold_replan_cases}",
            f"True accepts          : {metrics.true_accepts}",
            f"True rejects          : {metrics.true_rejects}",
            f"False accepts         : {metrics.false_accepts}",
            f"False acceptance rate : {metrics.false_acceptance_rate:.1%}",
            f"False rejects         : {metrics.false_rejects}",
            f"False rejection rate  : {metrics.false_rejection_rate:.1%}",
            f"Errors                : {metrics.errors}",
            "",
            "Per-case agreement:",
        ]
    )
    for agreement in run.agreements:
        lines.append(
            f"- {agreement.case_id}: {agreement.agreement_rate:.1%} agreement; "
            f"PASS={agreement.pass_count}, REPLAN={agreement.replan_count}, "
            f"ERROR={agreement.error_count}"
        )
    lines.extend(
        [
            "",
            "This is a small diagnostic evaluation, not a benchmark-quality estimate.",
            f"Detailed log: {run.log_path}",
            f"Machine-readable results: {run.results_path}",
            "",
            "Detailed prompts, model responses, and per-case metadata were saved to:",
            str(run.log_path),
        ]
    )
    return "\n".join(lines)
