"""Regression checks for the longitudinal prompt-decomposition ablation."""

import json

import pytest

from data_analysis_agent.benchmark import DEFAULT_TASKS_ROOT
from data_analysis_agent.benchmark_tasks import BenchmarkTaskError, load_benchmark_task


def test_longitudinal_prompt_variants_keep_the_task_defining_rules() -> None:
    """Checklist: catch accidental omission while allowing different organization."""
    recipe = load_benchmark_task(
        DEFAULT_TASKS_ROOT, "longitudinal_treatment_response", "recipe"
    ).public.prompt.lower()
    requirements = load_benchmark_task(
        DEFAULT_TASKS_ROOT,
        "longitudinal_treatment_response",
        "requirements_only",
    ).public.prompt.lower()
    checklist = {
        "age range": ("18", "75"),
        "baseline window": ("day -14", "day -1"),
        "follow-up window": ("day 28", "day 42", "day 35"),
        "visit statuses": ("valid", "reviewed"),
        "sample SD": ("n - 1",),
        "sample SE": ("sqrt(n)",),
        "B-minus-A comparison": ("mean change", "arm a", "arm b"),
        "attrition keys": (
            "total_patients",
            "invalid_or_missing_visit_rows_excluded",
        ),
        "selected-pair requirement": ("selected_pairs", "baseline", "follow-up"),
        "three-decimal rounding": ("three decimal",),
    }
    for name, terms in checklist.items():
        for term in terms:
            assert term in recipe, f"recipe omitted {name}: {term}"
            assert term in requirements, f"requirements-only omitted {name}: {term}"


def test_loader_rejects_unsafe_declared_prompt_path(tmp_path) -> None:
    public = tmp_path / "unsafe_prompt" / "public"
    public.mkdir(parents=True)
    (public / "task.json").write_text(
        json.dumps(
            {
                "default_prompt_variant": "unsafe",
                "prompt_variants": {"unsafe": "../private/reference.json"},
                "answer_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkTaskError, match="Unsafe prompt path"):
        load_benchmark_task(tmp_path, "unsafe_prompt")
