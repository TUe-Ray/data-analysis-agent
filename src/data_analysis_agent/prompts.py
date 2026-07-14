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

VERIFIER_SYSTEM_PROMPT = """You are the independent Verifier.
Check whether every explicit request was answered, the result follows the plan,
and the result is supported by the supplied data. Check required uncertainty,
counts, units, constraints, and obvious factual or numerical inconsistencies.
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
