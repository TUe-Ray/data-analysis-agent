from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "private/reference.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("task_grader", ROOT / "private/grader.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def require_fail(name: str, candidate: dict) -> None:
    result = module.grade(candidate)
    assert not result["passed"], f"mutation unexpectedly passed: {name}"


def main() -> int:
    assert module.grade(REFERENCE)["passed"]

    legacy = deepcopy(REFERENCE)
    pair = next(row for row in legacy["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "ALPHA-002")
    pair["followup_visit_record_id"] = "A02-FU-LEGACY"
    require_fail("legacy_alpha_followup", legacy)

    calibration = deepcopy(REFERENCE)
    pair = next(row for row in calibration["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "ALPHA-004")
    pair["baseline_harmonized_value"] = round(pair["baseline_harmonized_value"] / 1.08, 3)
    require_fail("omitted_alpha_calibration", calibration)

    crosswalk = deepcopy(REFERENCE)
    pair = next(row for row in crosswalk["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "BETA-001")
    pair["analysis_subject_id"] = "BETA-002"
    require_fail("beta_direct_join", crosswalk)

    replicate = deepcopy(REFERENCE)
    pair = next(row for row in replicate["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "BETA-004")
    pair["baseline_assay_record_ids"] = pair["baseline_assay_record_ids"][:1]
    require_fail("beta_selected_one_replicate", replicate)

    pooling = deepcopy(REFERENCE)
    comps = pooling["key_results"]["study_between_arm_comparisons"]
    pooling["key_results"]["pooled_comparison"]["pooled_difference_in_mean_change_b_minus_a"] = round(
        (comps["alpha"]["difference_in_mean_change_b_minus_a"] + comps["beta"]["difference_in_mean_change_b_minus_a"]) / 2,
        3,
    )
    require_fail("simple_average_pooling", pooling)

    se = deepcopy(REFERENCE)
    se["key_results"]["study_statistics"]["alpha"]["A"]["sample_se_change"] = se["key_results"]["study_statistics"]["alpha"]["A"]["sample_sd_change"]
    require_fail("sd_se_confusion", se)

    direction = deepcopy(REFERENCE)
    direction["key_results"]["study_between_arm_comparisons"]["alpha"]["difference_in_mean_change_b_minus_a"] *= -1
    require_fail("reversed_direction", direction)

    attrition = deepcopy(REFERENCE)
    attrition["key_results"]["study_attrition"]["beta"]["complete_pairs"] += 1
    require_fail("inconsistent_attrition", attrition)

    print("reference passed; 8 adversarial mutations rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
