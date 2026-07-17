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
explain hidden reasoning. Required outputs must be externally verifiable
JSON-compatible facts (for example counts, finite statistics, or concise status
facts) or explicitly declared analysis artifacts. Never require an in-memory
DataFrame, Series, array, or other Python object to appear in a goal result; a
tabular intermediate belongs in a declared artifact after its producing goal is
verified. A scientific replan returns only a replacement suffix beginning at the
failed unfinished goal; the runtime preserves completed definitions, results, and
approved artifacts exactly. Never reuse a retained completed goal_id for a new
definition.
When a revised goal needs a prior approved artifact, explicitly list its producer
goal_id in depends_on; artifacts are not otherwise available. Dependencies must be
declared in depends_on rather than inferred from goal prose. A goal receives
JSON results from all declared dependencies and their upstream dependencies.
When the public context contains a base specification, one or more amendments, and
document-precedence metadata, begin with a compact specification-reconciliation
goal. Its JSON-compatible result must state the effective amended rules, field and
identifier mappings, and applied precedence, remain externally verifiable, and
contain no calculated study outcomes. Every later goal that interprets scientific
rules or mappings must depend transitively on that reconciliation goal so its
verified GoalResult is the primary rule contract. Do not hard-code a particular
goal_id for this pattern. Make the contract operationally sufficient: include
exact physical coded values alongside semantic meanings, physical columns and join
keys, canonical identifiers, scientific record-identity fields (distinguishing
them from technical row keys), selection windows and tie-breaks, exclusion
boundaries, and statistical definitions. Declare effective specification, field
mappings, and document precedence as separate required outputs.
Design attrition goals in the task's declared sequential order. A removal count is
the number removed from the cohort remaining at that exact stage, so categories
must be mutually exclusive. Never place a post-selection exclusion or its count in
a goal that runs before the baseline/follow-up selection it depends on.
When a task combines multiple studies, sites, or source systems, a single
cohort-producing goal may apply their distinct rules provided its normalized
selected-pairs artifact keeps the systems separately auditable and carries their
separate attrition and data-quality facts. Downstream statistics must use that
approved artifact rather than rebuilding any cohort from raw files.
When more than one downstream quantitative goal depends on a derived analysis
cohort, add a producing goal that publishes the complete cohort as a declared
tabular artifact with identifiers, arm/group, source selections, and derived
values. Each consumer must depend on that producer and use the approved artifact
as its sole cohort source; do not independently rebuild the cohort from raw files.
When a public answer schema is supplied, set final_output_goal_id to the last goal.
That final assembly goal's dependency closure must include every earlier goal and
produce one complete answer object matching every required public-schema section;
intermediate goals must
produce only facts or declared artifacts, never partial final-answer wrappers."""

PLANNER_REPAIR_SYSTEM_PROMPT = """Repair one structurally invalid Planner response.
Return exactly one JSON object matching the schema requested for the active mode.
For initial planning, that schema is HighLevelPlan. For a scientific replan, it is
SuffixReplan and contains only the replacement suffix: the runtime owns the
immutable retained prefix. Use unique goal_id values and declare dependencies only
through depends_on. Do not calculate results or add implementation details, code,
tools, or file paths. Do not remove required task outputs merely to make the
structure valid. Express required outputs as JSON-compatible facts or declared
analysis artifacts, never as in-memory DataFrames, Series, arrays, or other Python
objects. Repair only the supplied response's structure; do not perform another
scientific replan."""

EXECUTOR_SYSTEM_PROMPT = """You are the tactical Executor for one scientific goal.
Choose a trusted built-in capability when it directly satisfies the goal; otherwise
choose generated_python. Choose structured_result for a compact document-only
reconciliation goal that needs no raw-data computation or tabular artifact. Do not
change the goal, remove outputs or constraints,
judge scientific validity, or add unsupported conclusions. Return only one JSON
ExecutionStrategy with strategy, capability_name, arguments, and a short
concise_reason. Arguments must validate against the selected capability. For
generated_python or structured_result, capability_name must be null and arguments
must be {}."""

STRUCTURED_RESULT_SYSTEM_PROMPT = """Return a compact JSON-compatible scientific
result for the fixed document-reconciliation goal. Use only the supplied public
documents, codebooks, precedence metadata, goal contract, and verified upstream
facts. Do not write or execute Python, do not reopen Markdown at runtime, and do
not publish artifacts. Return only the required StructuredResult JSON object with
result and warnings."""

PYTHON_GENERATION_SYSTEM_PROMPT = """Generate one deterministic Python script for
the supplied fixed scientific goal. Return only the required PythonGeneration JSON
object with kind="python", code_lines, and a concise summary. Each code_lines item
is exactly one physical Python source line: it must not contain a newline or
carriage return. Do not include Python comments in code_lines. Use only the standard
library, pandas, numpy, or scipy when already installed. Read only the explicitly
allowed staged files and use their exact absolute paths as direct string literals.
The process current working directory is the assigned goal directory, so relative
output paths remain inside it. Do not write outside that directory, access the
network, environment variables, subprocesses, shells, or package installers, or
delete files. Dynamically constructed file paths may be rejected, including paths
built from __file__, os.path, environment values, loops, globbing, or function
parameters. Assign the authoritative result object to the fixed variable
__agent_result__ at module scope. Do not return it only from a function or print it
as the authoritative channel. Do not manually serialize it. It must be one
JSON-compatible object containing only standard JSON-compatible Python values; do
not place DataFrames, Series, arrays, NumPy scalars, timestamps, Paths, sets, or
custom objects in it. Write tabular intermediates as declared artifacts instead.
To request downstream handoff, include an "artifacts" list in __agent_result__; each
item must contain relative_name, description, and optional media_type, and must name
an eligible file written inside the current goal directory. Such files are available
only after the independent Verifier returns PASS. If the goal requires a tabular or
file artifact, merely writing the file is not enough: declare that file in the
__agent_result__["artifacts"] list in the same execution. Never omit a required
artifact declaration while returning only its scalar audit counts.
Approved tabular artifacts list their factual columns. Use only columns that are
listed; when an artifact lacks a field required for the goal, read an explicitly
allowed staged input containing that field or an explicitly declared prerequisite
artifact. Do not probe for fields by raising a KeyError.
A verifier-approved artifact is the authoritative output of its producer goal. Do
not reapply an upstream validity, eligibility, mapping, or deduplication filter in
a downstream goal unless the current goal explicitly requires that audit; doing so
with semantic labels where physical codes are stored can silently remove valid
rows. When exact logical duplicate fields are supplied, exclude technical row keys
from that identity and deduplicate on exactly the declared scientific fields.
When a declared prerequisite artifact contains the derived cohort required for the
goal, use that artifact as the only cohort source. Do not reconstruct the cohort
from raw staged files or combine rows from separately recomputed versions; this
would make downstream attrition, pair listings, and statistics incompatible.
Never call drop_duplicates without the explicit scientific identity subset when
the specification distinguishes logical duplicates from technical rows. Preserve
the specification's declared stage order: when logical deduplication precedes
validity filtering, reconstruct the joined rows, deduplicate on the complete
scientific identity, and only then filter invalid rows. In that case, compute the
duplicate count from joined rows minus deduplicated rows and the invalid count
from deduplicated rows minus valid rows. Do not join an eligibility-filtered
subject artifact before computing visit-level audit counts unless the specification
explicitly limits those counts to eligible subjects.
For a specification-reconciliation goal, the complete documents are already in
the factual input context. Do not rely on whitespace-sensitive or line-wrap-
sensitive regular expressions to recover governing rules from Markdown; derive the
compact contract directly from the supplied evidence or parse headings and prose
robustly across whitespace. The contract must distinguish semantic code meanings
from the exact physical values stored in data. Record both, using the codebook to
resolve every accepted physical code; downstream code must filter on physical
values, not semantic labels. A codebook row explicitly marked not accepted must
never appear in an accepted-values list. Its field mappings must cover coded eligibility
fields, physical join relations, canonical and source identifiers, and the exact
scientific logical-duplicate identity (excluding technical row keys). Its effective
rules must also retain selection windows and tie-breaks, exclusion boundaries, and
statistical definitions such as SD denominator, SE formula, and comparison
direction. Do not return a smaller descriptive summary that downstream code would
need to reinterpret from the source documents.
For sequential attrition, compute each category from the cohort remaining after
all earlier categories; never count all matching records globally or copy a
preliminary overlapping count into the final result. Before returning, verify that
the declared eligible cohort equals the sum of sequential removals plus the final
complete cohort, and that arm counts sum to complete pairs.
Preserve upstream attrition denominators from dependency GoalResults. In
particular, never infer total input subjects from an eligibility-filtered artifact
or subtract upstream exclusions a second time. Every attrition count must be a
nonnegative integer.
When a dependency GoalResult already supplies a named attrition count that the
current goal must report, preserve that exact verified value. Do not recompute or
replace it from raw inputs; report a conflict explicitly if the prerequisite
artifact cannot support the required downstream calculation.
Keep source-system identifiers and canonical analysis identifiers distinct. When
an event table uses a source identifier, join it through the documented source
identifier mapping before comparing it with canonical selected records; never
join a source identifier directly to a canonical patient or analysis identifier.
Never put summary text, JSON framing, Markdown fences, or explanation inside
code_lines."""

PYTHON_REPAIR_SYSTEM_PROMPT = """Repair one mechanically failing generated Python
script. Preserve the exact goal, required outputs, constraints, and scientific
method. Fix implementation only. Return only the required PythonRepair JSON object
with kind="python_repair", code_lines, a concise summary, and the
addressed_failure_category. Each code_lines item is exactly one physical source line
without newline or carriage-return characters. Do not include Python comments in
code_lines. Use only the stated libraries and exact absolute staged-file paths as
direct literals. The process current working directory is the assigned goal
directory, so relative output paths remain inside it. Assign a JSON-compatible
object to __agent_result__ at module scope; do not manually serialize or print that
result. Do not place DataFrames, Series, arrays, NumPy scalars, timestamps, Paths,
sets, or custom objects in it. If the failure is PythonPolicyError, do not construct
paths dynamically. Repair the typed deterministic diagnosis supplied by the user,
not merely the script in general. If a required list or table exceeds the result
contract limit, never truncate, slice, sample, call head, or discard rows to fit
the limit. Write the complete table as a declared artifact and return only its
compact manifest, counts, and other required scalar facts. Never put summary text,
JSON framing, Markdown
fences, or explanation inside code_lines. Do not repair a failed Markdown regular
expression by making the phrase still more specific or by dereferencing another
unguarded match. Parse headings and fields tolerantly, or derive the compact rule
from the supplied factual context, and always handle an absent optional match."""

PYTHON_RESULT_SKELETON = """Minimal accepted pattern (replace paths and fields as
needed):
import pandas as pd

patients_df = pd.read_csv("/exact/staged/patients.csv")
visits_df = pd.read_csv("/exact/staged/visits.csv")
exclusions_df = pd.read_csv("/exact/staged/exclusions.csv")

__agent_result__ = {
    "patient_rows": int(len(patients_df)),
    "visit_rows": int(len(visits_df)),
    "exclusion_rows": int(len(exclusions_df)),
}
"""

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
   JSON numbers do not preserve insignificant trailing zeros: for example, 0.34
   and 0.340 are the same JSON numeric value. Never request a scientific replan
   solely to display trailing zeros when the value is already rounded to no more
   than the required decimal places.
9. A declared artifact is evidence only for its factual manifest. Do not PASS an
   artifact-dependent result if its listed columns cannot support the current goal
   or explicitly stated downstream data requirement.
10. For specification reconciliation, validate amendment precedence, accepted
    semantic values and their physical codes, analysis windows and targets,
    exclusion boundaries, statistical definitions, identifier/join mappings, and
    logical-record identity fields against the supplied documents. Reject a
    reconciliation result that is not operationally sufficient for downstream
    code or omits its declared effective-specification, mapping, or precedence
    contract.
11. When a verified dependency contains an effective specification, reject or
    replan any downstream rule, mapping, or boundary inconsistent with that
    contract and name the conflict precisely. Do not silently fall back to an
    obsolete base-document rule.
12. Attrition categories are sequential removals, not overlapping global event
    counts. Check the conservation identity from the reported fields and reject
    any result whose sequential removals plus final cohort do not equal the
    eligible cohort.

Return PASS only when no material issue remains. Return RETRY_GOAL when the fixed
goal can be regenerated with the same dependencies; return REPLAN only for a plan
or dependency-contract defect. Classify the issue as none, implementation, result,
artifact_handoff, dependency_contract, plan_contract, or evidence. Feedback must be
concise, specific, and actionable. Return only valid JSON."""

VERIFIER_REPAIR_PROMPT = """Your previous response was not valid for the required
JSON schema. Return only one valid JSON object with decision set to PASS,
RETRY_GOAL, or REPLAN, issue_classification, and a non-empty feedback string."""


def build_planner_messages(
    *,
    question: str,
    input_context: str,
    replan_count: int,
    verification_feedback: str | None = None,
    previous_plan: dict[str, JsonValue] | None = None,
    completed_goal_results: list[dict[str, JsonValue]] | None = None,
    completed_goal_fingerprints: dict[str, str] | None = None,
    approved_artifacts: list[dict[str, JsonValue]] | None = None,
    current_goal_failure: dict[str, JsonValue] | None = None,
    answer_schema: dict[str, JsonValue] | None = None,
    required_output_paths: list[str] | None = None,
    max_plan_goals: int = 6,
) -> list[dict[str, str]]:
    """Build global planning context without Executor histories or hidden thought."""
    parts = [
        f"Question:\n{question}",
        f"Input context and staged-file metadata:\n{input_context}",
        f"Current replan count: {replan_count}",
        f"Maximum high-level goals (including final assembly): {max_plan_goals}",
    ]
    if answer_schema:
        parts.extend(
            [
                "Exact public answer schema:\n"
                + json.dumps(answer_schema, ensure_ascii=False),
                "Required public output paths that final assembly must cover:\n"
                + json.dumps(required_output_paths or [], ensure_ascii=False),
            ]
        )
    if previous_plan is not None:
        parts.append(f"Previous high-level plan:\n{json.dumps(previous_plan)}")
    if completed_goal_results:
        parts.append(
            "Completed goal summaries:\n"
            + json.dumps(completed_goal_results, ensure_ascii=False)
        )
    if completed_goal_fingerprints:
        parts.append(
            "Completed goal definition fingerprints (these goals must be retained "
            "unchanged in the full revised plan):\n"
            + json.dumps(completed_goal_fingerprints, ensure_ascii=False)
        )
    if approved_artifacts:
        parts.append(
            "Verifier-approved artifacts (a consumer must explicitly include the "
            "producer_goal_id in depends_on):\n"
            + json.dumps(approved_artifacts, ensure_ascii=False)
        )
    if current_goal_failure is not None:
        parts.append(f"Current goal failure:\n{json.dumps(current_goal_failure)}")
    if verification_feedback is not None:
        parts.append(f"Verifier feedback:\n{verification_feedback}")
    if previous_plan is not None:
        parts.append(
            "Scientific replan requirement: return a SuffixReplan JSON object: "
            '{"replace_from_goal_id":"unfinished goal id","replacement_goals":['
            '...],"final_output_goal_id":"...","reason":"..."}. Do not repeat '
            "the retained prefix."
        )
    if previous_plan is None:
        parts.append(
            "HighLevelPlan JSON schema summary:\n"
            '{"scientific_objective":"string","goals":[{"goal_id":"string",'
            '"objective":"string","required_outputs":["string"],'
            '"constraints":["string"],"success_criteria":["string"],'
            '"depends_on":["earlier_goal_id"]}],'
            '"final_output_goal_id":"last_goal_id",'
            '"invalidate_from_goal_id":null}'
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
    previous_plan: dict[str, JsonValue] | None = None,
    completed_goal_fingerprints: dict[str, str] | None = None,
    completed_goal_definitions: list[dict[str, JsonValue]] | None = None,
    approved_artifacts: list[dict[str, JsonValue]] | None = None,
    answer_schema: dict[str, JsonValue] | None = None,
    required_output_paths: list[str] | None = None,
    max_plan_goals: int = 6,
    planner_mode: str = "initial",
) -> list[dict[str, str]]:
    """Build a compact structural-only repair request for Planner output."""
    initial_schema_summary = (
        '{"scientific_objective":"string","goals":[{"goal_id":"string",'
        '"objective":"string","required_outputs":["string"],'
        '"constraints":["string"],"success_criteria":["string"],'
        '"depends_on":["earlier_goal_id"]}],'
        '"final_output_goal_id":"last_goal_id",'
        '"invalidate_from_goal_id":null}'
    )
    suffix_schema_summary = (
        '{"replace_from_goal_id":"unfinished_goal_id",'
        '"replacement_goals":[{"goal_id":"string","objective":"string",'
        '"required_outputs":["string"],"constraints":["string"],'
        '"success_criteria":["string"],"depends_on":["earlier_goal_id"]}],'
        '"final_output_goal_id":"last_goal_id","reason":"string"}'
    )
    is_replan = planner_mode == "scientific_replan"
    schema_summary = suffix_schema_summary if is_replan else initial_schema_summary
    return [
        {"role": "system", "content": PLANNER_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original user question:\n{question}\n\n"
                f"Invalid Planner response:\n{invalid_response}\n\n"
                f"Deterministic validation error:\n{validation_error}\n\n"
                + (
                    "Immutable retained prefix summary; do not repeat or rewrite it "
                    "in the suffix response:\n"
                    f"{json.dumps(previous_plan)}\n\n"
                    if previous_plan is not None
                    else ""
                )
                + (
                    "Immutable completed goal definitions are runtime-owned context; "
                    "do not include them in the suffix:\n"
                    f"{json.dumps(completed_goal_definitions)}\n\n"
                    if completed_goal_definitions
                    else ""
                )
                + (
                    "Completed immutable goal fingerprints:\n"
                    f"{json.dumps(completed_goal_fingerprints)}\n\n"
                    if completed_goal_fingerprints
                    else ""
                )
                + (
                    "Approved artifacts; consumers must depend on their producer:\n"
                    f"{json.dumps(approved_artifacts)}\n\n"
                    if approved_artifacts
                    else ""
                )
                + f"Active planner mode: {planner_mode}\n"
                + f"Required JSON schema summary:\n{schema_summary}\n\n"
                + f"Maximum goals: {max_plan_goals}\n\n"
                + (
                    "Exact public answer schema:\n"
                    f"{json.dumps(answer_schema, ensure_ascii=False)}\n\n"
                    "Required final output paths:\n"
                    f"{json.dumps(required_output_paths or [])}\n\n"
                    if answer_schema
                    else ""
                )
                + "Dependency rule: every depends_on item must exactly match an "
                "existing earlier goal_id; forward and missing dependencies are "
                "invalid. Return only the schema requested for the active mode."
            ),
        },
    ]


def build_executor_strategy_repair_messages(
    *, invalid_response: str, validation_error: str
) -> list[dict[str, str]]:
    """Request one structural-only correction of an Executor strategy."""
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Repair only the malformed ExecutionStrategy JSON below. Return "
                "one object with exactly strategy, capability_name, arguments, "
                "and concise_reason. Do not revise the scientific goal.\n\n"
                f"Invalid response:\n{invalid_response}\n\n"
                f"Validation error:\n{validation_error}"
            ),
        },
    ]


def build_structured_result_messages(
    *,
    current_goal: dict[str, JsonValue],
    input_context: str,
    completed_goal_results: list[dict[str, JsonValue]],
    verification_feedback: str | None = None,
    previous_attempt_result: dict[str, JsonValue] | None = None,
) -> list[dict[str, str]]:
    """Build a no-code prompt for compact document reasoning results."""
    parts = [
        f"Current IntermediateGoal:\n{json.dumps(current_goal)}",
        f"Public specification context:\n{input_context}",
        "Verified prerequisite GoalResults:\n"
        + json.dumps(completed_goal_results, ensure_ascii=False),
    ]
    if verification_feedback:
        parts.append(f"Exact verifier feedback:\n{verification_feedback}")
    if previous_attempt_result is not None:
        parts.append(
            "Rejected previous result; retain sound facts and correct every issue:\n"
            + json.dumps(previous_attempt_result, ensure_ascii=False)
        )
    return [
        {"role": "system", "content": STRUCTURED_RESULT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
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
    approved_artifacts: list[dict[str, object]] | None = None,
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
            "Verifier-approved prerequisite artifacts (producer, exact path, "
            "description, media type):\n" + json.dumps(approved_artifacts or []),
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
    input_context: str = "",
    approved_artifacts: list[dict[str, object]] | None = None,
    result_schema: dict[str, JsonValue] | None = None,
    verification_feedback: str | None = None,
    previous_attempt_result: dict[str, JsonValue] | None = None,
    previous_attempt_code: str | None = None,
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
                f"Staged input schema and factual context:\n{input_context}\n\n"
                + (
                    "Verifier correction from the preceding scientific replan; "
                    "the regenerated implementation must preserve correct prior "
                    "facts and make every requested correction without changing "
                    "the approved goal:\n"
                    f"{verification_feedback}\n\n"
                    if verification_feedback
                    else ""
                )
                + (
                    "Rejected result from the immediately preceding attempt. "
                    "Use it only as a concrete edit target; do not treat it as "
                    "approved evidence, and correct every verifier issue:\n"
                    f"{json.dumps(previous_attempt_result, ensure_ascii=False)}\n\n"
                    if previous_attempt_result is not None
                    else ""
                )
                + (
                    "Previous source from that rejected attempt. Regenerate this "
                    "complete script as a focused correction: retain sound work, "
                    "make every verifier-requested correction, and do not replace "
                    "it with a less complete analysis:\n"
                    f"{previous_attempt_code}\n\n"
                    if previous_attempt_code is not None
                    else ""
                )
                + "If dependency_goal_results.json is listed, it is the only "
                "runner-created file containing prerequisite results. Read "
                "that exact path with json.load when prior result values are "
                'needed. Its exact shape is {"goal_results":[{"goal_id":"...",'
                '"required_outputs":["..."],"result":{...}}]}; read '
                'payload["goal_results"] and find the '
                "entry with the required goal_id. No prerequisite Python variables "
                "are preloaded, and you "
                "must never invent or read files such as "
                "goals/<goal_id>/results.json.\n\n"
                "Completed prerequisite results:\n"
                f"{json.dumps(completed_goal_results)}\n\n"
                "Verifier-approved prerequisite artifacts:\n"
                f"{json.dumps(approved_artifacts or [])}\n\n"
                "Exact required result schema for this goal (null means only the "
                "goal contract applies):\n"
                f"{json.dumps(result_schema, ensure_ascii=False)}\n\n"
                f"Assigned goal directory and process cwd:\n{goal_directory}\n\n"
                "Assign exactly one JSON-compatible object to __agent_result__. "
                "The trusted runner serializes it with allow_nan=False.\n\n"
                + PYTHON_RESULT_SKELETON
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
    failure_fingerprint: str | None = None,
    staged_file_paths: list[str],
    goal_directory: str,
    input_context: str = "",
    completed_goal_results: list[dict[str, JsonValue]] | None = None,
    repair_history: list[dict[str, object]] | None = None,
    approved_artifacts: list[dict[str, object]] | None = None,
    result_schema: dict[str, JsonValue] | None = None,
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
                "Normalized concrete failure fingerprint:\n"
                f"{failure_fingerprint or failure_category}\n\n"
                f"stdout:\n{stdout}\n\n"
                f"stderr:\n{stderr}\n\n"
                f"Execution error:\n{error or 'none'}\n\n"
                "Allowed libraries: Python standard library, pandas, numpy, scipy.\n"
                f"Allowed files:\n{json.dumps(staged_file_paths)}\n\n"
                f"Staged input schema and factual context:\n{input_context}\n\n"
                "If dependency_goal_results.json is listed, read that exact file "
                "for prerequisite result values. Its exact shape is "
                '{"goal_results":[{"goal_id":"...","required_outputs":["..."],'
                '"result":{...}}]}; use payload["goal_results"] and find the '
                "matching goal_id. Check required_outputs before assuming a result "
                "contains a value. "
                "No prerequisite "
                "Python "
                "variables are preloaded. Never invent or read files such as "
                "goals/<goal_id>/results.json.\n\n"
                "Completed prerequisite results:\n"
                f"{json.dumps(completed_goal_results or [])}\n\n"
                "Verifier-approved prerequisite artifacts:\n"
                f"{json.dumps(approved_artifacts or [])}\n\n"
                "Exact required result schema for this goal:\n"
                f"{json.dumps(result_schema, ensure_ascii=False)}\n\n"
                f"Allowed output directory and process cwd:\n{goal_directory}\n\n"
                "Recent compact repair history:\n"
                f"{json.dumps(repair_history or [], ensure_ascii=False)}\n\n"
                "Change the implementation that caused this mechanical failure; do "
                "not change the scientific method, constraints, or required "
                "outputs.\n\n"
                "Assign the JSON-compatible result object to __agent_result__; the "
                "trusted runner owns serialization and stdout is not authoritative.\n\n"
                "Deterministic diagnosis to repair:\n"
                f"{error or failure_category}\n\n" + PYTHON_RESULT_SKELETON + "\n"
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
    pending_artifacts: list[dict[str, JsonValue]] | None = None,
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
        sections = [
            f"Original question:\n{question}",
            f"Scientific objective:\n{scientific_objective or ''}",
            f"Current IntermediateGoal:\n{json.dumps(current_goal)}",
            f"Execution strategy summary:\n{json.dumps(strategy or {})}",
            f"Factual execution result:\n{execution_result}",
            f"Warnings:\n{json.dumps(warnings or [])}",
            "Pending analysis artifact manifests (CSV headers and row counts are "
            "factual). Return REPLAN when a declared artifact lacks fields needed "
            "for the current goal or its stated downstream output:\n"
            + json.dumps(pending_artifacts or []),
            "Relevant prior GoalResults:\n" + json.dumps(prior_goal_results or []),
        ]
        if input_context:
            sections.insert(2, f"Public input context and evidence:\n{input_context}")
        body = "\n\n".join(sections)
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]
