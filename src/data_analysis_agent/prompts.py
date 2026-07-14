"""Role-specific prompts and isolated message construction."""

from __future__ import annotations

PLANNER_SYSTEM_PROMPT = """You are the Planner for a scientific analysis.
Create a concise numbered plan that identifies every explicitly requested output.
Do not execute calculations. When verifier feedback is supplied, revise the plan
to address it. Do not explain hidden reasoning. Return only the plan."""

EXECUTOR_SYSTEM_PROMPT = """You are the Executor for a scientific analysis.
Follow the supplied plan and use only the supplied input context. Answer every
requested item and briefly report assumptions or omitted values. Do not evaluate
whether your own result is valid. Return only the execution result."""

VERIFIER_SYSTEM_PROMPT = """You are the independent scientific Verifier.
Judge only the supplied question, input context, plan, and execution result.

Apply this rubric:
1. Completeness: every explicitly requested output must be present.
2. Constraint compliance: required missing-value handling and prohibitions such as
   no imputation must be followed.
3. Data support: every result and claim must be supported by the supplied data.
4. Numerical consistency: counts, means, and sample standard errors must be
   consistent; do not confuse standard deviation with standard error.
5. Plan alignment: the analysis must be consistent with the supplied procedure.
   Do not require every procedural step to be narrated in the final result unless
   the user explicitly requests that explanation. A concise result may demonstrate
   constraint compliance through values that are consistent with the required
   procedure.
6. Claim discipline: reject unsupported causal, significance, or trend claims.
7. Materiality: accept reasonable rounding and concise but complete answers.

Return PASS only when no material error, missing requested output, violated
constraint, or unsupported scientific claim remains. Return REPLAN when a
material correction is needed. Feedback must be concise, specific, and actionable.
Return only valid JSON with exactly this shape:
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
) -> list[dict[str, str]]:
    """Build a Planner context containing only inputs allowed for that role."""
    parts = [
        f"Question:\n{question}",
        f"Input context:\n{input_context}",
        f"Current replan count: {replan_count}",
    ]
    if verification_feedback is not None:
        parts.append(f"Verifier feedback:\n{verification_feedback}")
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_executor_messages(
    *, question: str, input_context: str, plan: str
) -> list[dict[str, str]]:
    """Build an Executor context containing only the question, data, and plan."""
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Input context:\n{input_context}\n\n"
                f"Plan:\n{plan}"
            ),
        },
    ]


def build_verifier_messages(
    *, question: str, input_context: str, plan: str, execution_result: str
) -> list[dict[str, str]]:
    """Build an evidence-oriented Verifier context with no role histories."""
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Input context:\n{input_context}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Execution result:\n{execution_result}"
            ),
        },
    ]
