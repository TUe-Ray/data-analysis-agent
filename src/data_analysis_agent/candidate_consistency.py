"""Deterministic candidate-internal checks for the independent final checker."""

from __future__ import annotations

import math
import statistics

from pydantic import JsonValue


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _close(left: object, right: object, *, tolerance: float = 0.0015) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return (
        left_number is not None
        and right_number is not None
        and abs(left_number - right_number) <= tolerance
    )


def build_candidate_consistency_evidence(
    candidate: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Recompute only identities contained in a candidate; never alter its values."""
    key_results = candidate.get("key_results", {})
    if not isinstance(key_results, dict):
        key_results = {}
    attrition = key_results.get("attrition", {})
    if not isinstance(attrition, dict):
        attrition = {}
    arm_statistics = key_results.get("arm_statistics", {})
    if not isinstance(arm_statistics, dict):
        arm_statistics = {}
    raw_pairs = key_results.get("selected_pairs", [])
    pairs = raw_pairs if isinstance(raw_pairs, list) else []

    pair_change_mismatches: list[str] = []
    derived_changes: dict[str, list[float]] = {"A": [], "B": []}
    pair_counts = {"A": 0, "B": 0}
    for index, raw_pair in enumerate(pairs):
        if not isinstance(raw_pair, dict):
            pair_change_mismatches.append(f"index:{index}:not_an_object")
            continue
        patient_id = str(raw_pair.get("patient_id", f"index:{index}"))
        arm = raw_pair.get("arm")
        baseline = _number(raw_pair.get("baseline_value"))
        followup = _number(raw_pair.get("followup_value"))
        reported_change = raw_pair.get("change")
        if arm in pair_counts:
            pair_counts[str(arm)] += 1
        if baseline is None or followup is None:
            pair_change_mismatches.append(f"{patient_id}:non_numeric_pair")
            continue
        derived = followup - baseline
        if not _close(derived, reported_change):
            pair_change_mismatches.append(patient_id)
        if arm in derived_changes:
            derived_changes[str(arm)].append(derived)

    recomputed_arm_statistics: dict[str, JsonValue] = {}
    aggregate_checks: dict[str, JsonValue] = {}
    raw_mean_changes: dict[str, float] = {}
    for arm in ("A", "B"):
        changes = derived_changes[arm]
        reported = arm_statistics.get(arm, {})
        if not isinstance(reported, dict):
            reported = {}
        if changes:
            mean_change = statistics.mean(changes)
            sample_sd = statistics.stdev(changes) if len(changes) > 1 else None
            sample_se = (
                sample_sd / math.sqrt(len(changes)) if sample_sd is not None else None
            )
            raw_mean_changes[arm] = mean_change
        else:
            mean_change = None
            sample_sd = None
            sample_se = None
        recomputed_arm_statistics[arm] = {
            "n": len(changes),
            "mean_change": mean_change,
            "sample_sd_change": sample_sd,
            "sample_se_change": sample_se,
        }
        aggregate_checks[f"arm_{arm.lower()}_n_matches_pairs"] = reported.get(
            "n"
        ) == len(changes)
        aggregate_checks[f"arm_{arm.lower()}_mean_change_matches_pairs"] = _close(
            mean_change, reported.get("mean_change")
        )
        aggregate_checks[f"arm_{arm.lower()}_sample_sd_matches_pairs"] = _close(
            sample_sd, reported.get("sample_sd_change")
        )
        aggregate_checks[f"arm_{arm.lower()}_sample_se_matches_pairs"] = _close(
            sample_se, reported.get("sample_se_change")
        )

    reported_comparison = key_results.get("between_arm_comparison", {})
    if not isinstance(reported_comparison, dict):
        reported_comparison = {}
    recomputed_difference = (
        raw_mean_changes["B"] - raw_mean_changes["A"]
        if set(raw_mean_changes) == {"A", "B"}
        else None
    )
    aggregate_checks["b_minus_a_matches_pairs"] = _close(
        recomputed_difference,
        reported_comparison.get("difference_in_mean_change_b_minus_a"),
    )
    attrition_checks = {
        "complete_pairs_matches_selected_pairs": attrition.get("complete_pairs")
        == len(pairs),
        "complete_pair_arm_counts_match_selected_pairs": (
            attrition.get("complete_pairs_arm_a") == pair_counts["A"]
            and attrition.get("complete_pairs_arm_b") == pair_counts["B"]
        ),
        "complete_pair_arm_counts_sum_to_total": (
            _number(attrition.get("complete_pairs_arm_a")) is not None
            and _number(attrition.get("complete_pairs_arm_b")) is not None
            and _number(attrition.get("complete_pairs")) is not None
            and float(attrition["complete_pairs_arm_a"])
            + float(attrition["complete_pairs_arm_b"])
            == float(attrition["complete_pairs"])
        ),
        "basic_attrition_identity_holds": (
            _number(attrition.get("total_patients")) is not None
            and _number(attrition.get("basic_ineligible")) is not None
            and _number(attrition.get("eligible_after_basic_checks")) is not None
            and float(attrition["total_patients"])
            - float(attrition["basic_ineligible"])
            == float(attrition["eligible_after_basic_checks"])
        ),
        "sequential_attrition_conservation_holds": (
            all(
                _number(attrition.get(key)) is not None
                for key in (
                    "eligible_after_basic_checks",
                    "excluded_pre_start",
                    "no_valid_baseline",
                    "no_valid_followup",
                    "excluded_post_start_before_or_on_followup",
                    "complete_pairs",
                )
            )
            and float(attrition["eligible_after_basic_checks"])
            == sum(
                float(attrition[key])
                for key in (
                    "excluded_pre_start",
                    "no_valid_baseline",
                    "no_valid_followup",
                    "excluded_post_start_before_or_on_followup",
                    "complete_pairs",
                )
            )
        ),
    }
    return {
        "label": "candidate-internal consistency evidence",
        "scope_note": (
            "Derived only from the candidate; not an independent reconstruction "
            "from raw public data and not a replacement for private grading."
        ),
        "selected_pair_count": len(pairs),
        "selected_pair_counts_by_arm": pair_counts,
        "pair_change_mismatches": pair_change_mismatches,
        "recomputed_arm_change_statistics": recomputed_arm_statistics,
        "recomputed_b_minus_a": recomputed_difference,
        "aggregate_checks": aggregate_checks,
        "attrition_identity_checks": attrition_checks,
    }
