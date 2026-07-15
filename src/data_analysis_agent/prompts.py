"""Role-specific prompts with deliberately isolated factual contexts."""

from __future__ import annotations

import json

from pydantic import JsonValue

PLANNER_SYSTEM_PROMPT = """You are the Planner for a scientific analysis.
Return only one JSON object matching the supplied HighLevelPlan schema. Define the
global scientific objective and a small ordered list of high-level intermediate
goals. Include required outputs, constraints, concise success criteria, and only
necessary dependencies. Do not calculate results or prescribe tool names, Python
functions, imports, file paths, retries, low-level implementation steps, or
algorithm parameters unless the user explicitly requires a parameter. On replan,
revise high-level goals using the factual failure and verifier feedback. Do not
explain hidden reasoning."""

PLANNER_REPAIR_SYSTEM_PROMPT = """Repair one structurally invalid scientific
HighLevelPlan response. Return exactly one JSON object matching the required
HighLevelPlan schema. Preserve the original scientific task, required coverage,
and valid goals whenever possible. Use unique goal_id values. Each dependency
must name an existing goal_id that appears earlier in the goals list. Do not
calculate results or add implementation details, code, tools, or file paths. Do
not remove required task outputs merely to make the structure valid."""

EXECUTOR_SYSTEM_PROMPT = """You are the tactical Executor for one scientific goal.
Choose a trusted built-in capability when it directly satisfies the goal; otherwise
choose generated_python. Do not change the goal, remove outputs or constraints,
judge scientific validity, or add unsupported conclusions. Return only one JSON
ExecutionStrategy with strategy, capability_name, arguments, and a short
concise_reason. Arguments must validate against the selected capability. For
generated_python, capability_name must be null and arguments must be {}."""

PYTHON_GENERATION_SYSTEM_PROMPT = """Generate one deterministic Python script for
the supplied fixed scientific goal. Return only the required PythonGeneration JSON
object with kind="python", complete source in code, and a concise summary. Use only
the standard library, pandas, numpy, or scipy when already installed. Read only the
explicitly allowed staged files and use their exact absolute paths as direct string
literals. The process current working directory is the assigned goal directory, so
relative output paths remain inside it. Do not write outside that directory, access
the network, environment variables, subprocesses, shells, or package installers,
or delete files. Dynamically constructed file paths may be rejected, including
paths built from __file__, os.path, environment values, loops, globbing, or
function parameters. Assign the authoritative result object to the fixed variable
__agent_result__. Do not manually serialize or print the authoritative result;
debug stdout is allowed. Never put summary text, JSON framing, Markdown fences, or
explanation inside code."""

PYTHON_REPAIR_SYSTEM_PROMPT = """Repair one mechanically failing generated Python
script. Preserve the exact goal, required outputs, constraints, and scientific
method. Fix implementation only. Return only the required PythonRepair JSON object
with kind="python_repair", complete source in code, a concise summary, and the
addressed_failure_category. Use only the stated libraries and exact absolute staged
file paths as direct literals. The process current working directory is the assigned
goal directory, so relative output paths remain inside it. Assign the authoritative
result object to __agent_result__; do not manually serialize or print that result.
If the failure is PythonPolicyError, do not construct paths dynamically. Never put
summary text, JSON framing, Markdown fences, or explanation inside code."""

VERIFIER_SYSTEM_PROMPT = """You are the independent scientific Verifier.
Judge only the supplied original question, any supplied input context or scientific
objective, current plan or goal, execution strategy summary, factual execution
result, warnings, and relevant prior goal results.

Apply this rubric:
1. Required outputs are present.
2. Explicit goal constraints are respected.
3. Claims and values are supported by the execution output and supplied data.
4. Counts, means, and sample standard errors are numerically consistent; sample
   standard deviation must not be confused with standard error.
5. Generated Python actually completed successfully.
6. No unsupported causal, significance, trend, or other scientific conclusion was
   added.
7. The result contributes to the original scientific objective and follows the
   supplied plan without requiring every procedural step to be narrated.
8. Accept reasonable rounding, but reject material omissions or errors.

Return PASS only when no material issue remains. Return REPLAN when correction is
needed. Feedback must be concise, specific, and actionable. Return only valid JSON
with exactly this shape:
{"decision": "PASS" or "REPLAN", "feedback": "concise explanation"}"""

VERIFIER_REPAIR_PROMPT = """Your previous response was not valid for the required
JSON schema. Return only one valid JSON object with decision set to PASS or REPLAN
and a non-empty feedback string."""


def build_planner_messages(
    *,
    question: str,
    input_context: str,
    replan_count: int,
    verification_feedback: str | None = None,
    previous_plan: dict[str, JsonValue] | None = None,
    completed_goal_results: list[dict[str, JsonValue]] | None = None,
    current_goal_failure: dict[str, JsonValue] | None = None,
) -> list[dict[str, str]]:
    """Build global planning context without Executor histories or hidden thought."""
    parts = [
        f"Question:\n{question}",
        f"Input context and staged-file metadata:\n{input_context}",
        f"Current replan count: {replan_count}",
    ]
    if previous_plan is not None:
        parts.append(f"Previous high-level plan:\n{json.dumps(previous_plan)}")
    if completed_goal_results:
        parts.append(
            "Completed goal summaries:\n"
            + json.dumps(completed_goal_results, ensure_ascii=False)
        )
    if current_goal_failure is not None:
        parts.append(f"Current goal failure:\n{json.dumps(current_goal_failure)}")
    if verification_feedback is not None:
        parts.append(f"Verifier feedback:\n{verification_feedback}")
    parts.append(
        "HighLevelPlan JSON schema summary:\n"
        '{"scientific_objective":"string","goals":[{"goal_id":"string",'
        '"objective":"string","required_outputs":["string"],'
        '"constraints":["string"],"success_criteria":["string"],'
        '"depends_on":["earlier_goal_id"]}]}'
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_planner_repair_messages(
    *,
    question: str,
    invalid_response: str,
    validation_error: str,
) -> list[dict[str, str]]:
    """Build a compact structural-only repair request for Planner output."""
    schema_summary = (
        '{"scientific_objective":"string","goals":[{"goal_id":"string",'
        '"objective":"string","required_outputs":["string"],'
        '"constraints":["string"],"success_criteria":["string"],'
        '"depends_on":["earlier_goal_id"]}]}'
    )
    return [
        {"role": "system", "content": PLANNER_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original user question:\n{question}\n\n"
                f"Invalid Planner response:\n{invalid_response}\n\n"
                f"Deterministic validation error:\n{validation_error}\n\n"
                f"HighLevelPlan JSON schema summary:\n{schema_summary}\n\n"
                "Dependency rule: every depends_on item must exactly match an "
                "existing earlier goal_id; forward and missing dependencies are "
                "invalid."
            ),
        },
    ]


def build_executor_messages(
    *,
    question: str,
    input_context: str,
    plan: str | None = None,
    current_goal: dict[str, JsonValue] | None = None,
    completed_goal_results: list[dict[str, JsonValue]] | None = None,
    verification_feedback: str | None = None,
    capability_catalog: list[dict[str, JsonValue]] | None = None,
    staged_file_paths: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build a one-goal tactical context; ``plan`` retains V0 compatibility."""
    if current_goal is None:
        body = (
            f"Question:\n{question}\n\n"
            f"Input context:\n{input_context}\n\n"
            f"Plan:\n{plan or ''}"
        )
    else:
        parts = [
            f"Original question:\n{question}",
            f"Staged input context or metadata:\n{input_context}",
            f"Explicitly staged paths:\n{json.dumps(staged_file_paths or [])}",
            f"Current IntermediateGoal:\n{json.dumps(current_goal)}",
            "Completed prerequisite GoalResults:\n"
            + json.dumps(completed_goal_results or []),
            "Available capability catalog:\n" + json.dumps(capability_catalog or []),
        ]
        if verification_feedback:
            parts.append(f"Concise verifier feedback:\n{verification_feedback}")
        body = "\n\n".join(parts)
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]


def build_python_generation_messages(
    *,
    current_goal: dict[str, JsonValue],
    staged_file_paths: list[str],
    completed_goal_results: list[dict[str, JsonValue]],
    goal_directory: str,
) -> list[dict[str, str]]:
    """Supply generated-code creation only the current factual execution context."""
    return [
        {"role": "system", "content": PYTHON_GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Current IntermediateGoal:\n{json.dumps(current_goal)}\n\n"
                f"Allowed input files:\n{json.dumps(staged_file_paths)}\n\n"
                "Use those exact absolute paths directly as string literals when "
                "reading data.\n\n"
                "Completed prerequisite results:\n"
                f"{json.dumps(completed_goal_results)}\n\n"
                f"Assigned goal directory and process cwd:\n{goal_directory}\n\n"
                "Assign exactly one JSON-compatible object to __agent_result__. "
                "The trusted runner serializes it with allow_nan=False."
            ),
        },
    ]


def build_python_repair_messages(
    *,
    current_goal: dict[str, JsonValue],
    code: str,
    failure_category: str,
    stdout: str,
    stderr: str,
    error: str | None,
    staged_file_paths: list[str],
    goal_directory: str,
    repair_history: list[dict[str, object]] | None = None,
) -> list[dict[str, str]]:
    """Supply repair only the fixed goal, code, local failure, and allowlist."""
    return [
        {"role": "system", "content": PYTHON_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Current IntermediateGoal:\n{json.dumps(current_goal)}\n\n"
                f"Generated code:\n{code}\n\n"
                f"Typed failure category:\n{failure_category}\n\n"
                f"stdout:\n{stdout}\n\n"
                f"stderr:\n{stderr}\n\n"
                f"Execution error:\n{error or 'none'}\n\n"
                "Allowed libraries: Python standard library, pandas, numpy, scipy.\n"
                f"Allowed files:\n{json.dumps(staged_file_paths)}\n\n"
                f"Allowed output directory and process cwd:\n{goal_directory}\n\n"
                "Recent compact repair history:\n"
                f"{json.dumps(repair_history or [], ensure_ascii=False)}\n\n"
                "Change the implementation that caused this mechanical failure; do "
                "not change the scientific method, constraints, or required "
                "outputs.\n\n"
                "Assign the JSON-compatible result object to __agent_result__; the "
                "trusted runner owns serialization and stdout is not authoritative.\n\n"
                "If this is a PythonPolicyError, repair using only the exact "
                "absolute paths above as direct literals. Do not use __file__, "
                "os.path, environment values, globbing, loops, or function "
                "parameters to construct input paths."
            ),
        },
    ]


def build_verifier_messages(
    *,
    question: str,
    input_context: str | None = None,
    plan: str | None = None,
    execution_result: str,
    scientific_objective: str | None = None,
    current_goal: dict[str, JsonValue] | None = None,
    strategy: dict[str, JsonValue] | None = None,
    warnings: list[str] | None = None,
    prior_goal_results: list[dict[str, JsonValue]] | None = None,
) -> list[dict[str, str]]:
    """Build evidence-oriented verification context with no role histories."""
    if current_goal is None:
        body = (
            f"Question:\n{question}\n\n"
            f"Input context:\n{input_context or ''}\n\n"
            f"Plan:\n{plan or ''}\n\n"
            f"Execution result:\n{execution_result}"
        )
    else:
        body = "\n\n".join(
            [
                f"Original question:\n{question}",
                f"Scientific objective:\n{scientific_objective or ''}",
                f"Current IntermediateGoal:\n{json.dumps(current_goal)}",
                f"Execution strategy summary:\n{json.dumps(strategy or {})}",
                f"Factual execution result:\n{execution_result}",
                f"Warnings:\n{json.dumps(warnings or [])}",
                "Relevant prior GoalResults:\n" + json.dumps(prior_goal_results or []),
            ]
        )
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]
