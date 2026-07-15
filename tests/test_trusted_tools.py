from pathlib import Path

import pytest

from data_analysis_agent.trusted_tools import TrustedToolRegistry


def registry_for(path: Path) -> TrustedToolRegistry:
    return TrustedToolRegistry(allowed_roots=[path.parent], allowed_files=[path])


def test_registry_contains_only_the_three_builtin_tools(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    path.write_text("value\n1\n2\n", encoding="utf-8")
    registry = registry_for(path)

    assert registry.names == {
        "inspect_file",
        "profile_table",
        "compute_summary_statistics",
    }
    assert len(registry.catalog()) == 3


def test_inspect_and_profile_csv_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    path.write_text("id,value\na,10\nb,\nc,14\nc,14\n", encoding="utf-8")
    registry = registry_for(path)

    inspected = registry.execute("inspect_file", {"file_path": path.name})
    profile = registry.execute(
        "profile_table", {"file_path": path.name, "sample_rows": 2}
    )

    assert inspected.success
    assert inspected.output["row_count"] == 4
    assert inspected.output["column_names"] == ["id", "value"]
    assert profile.success
    assert profile.output["missing_count"] == {"id": 0, "value": 1}
    assert profile.output["duplicate_row_count"] == 1
    assert len(profile.output["preview"]) == 2


def test_summary_statistics_drop_missing_and_use_sample_definition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "values.csv"
    path.write_text('value\n10\n12\n""\n14\n16\n', encoding="utf-8")
    result = registry_for(path).execute(
        "compute_summary_statistics",
        {
            "file_path": path.name,
            "column": "value",
            "statistics": [
                "count",
                "mean",
                "sample_standard_deviation",
                "sample_standard_error",
                "quantiles",
            ],
            "drop_missing": True,
        },
    )

    assert result.success
    assert result.output["rows_used"] == 4
    assert result.output["missing_rows_excluded"] == 1
    values = result.output["statistics"]
    assert values["count"] == 4
    assert values["mean"] == 13
    assert values["sample_standard_deviation"] == pytest.approx(2.581988897)
    assert values["sample_standard_error"] == pytest.approx(1.290994449)


def test_summary_statistics_never_silently_impute(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    path.write_text('value\n1\n""\n2\n', encoding="utf-8")

    result = registry_for(path).execute(
        "compute_summary_statistics",
        {
            "file_path": path.name,
            "column": "value",
            "statistics": ["mean"],
            "drop_missing": False,
        },
    )

    assert not result.success
    assert "Missing values" in (result.error or "")


def test_sample_error_fails_with_fewer_than_two_values(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    path.write_text("value\n1\n", encoding="utf-8")

    result = registry_for(path).execute(
        "compute_summary_statistics",
        {
            "file_path": path.name,
            "column": "value",
            "statistics": ["sample_standard_error"],
            "drop_missing": True,
        },
    )

    assert not result.success
    assert "At least two" in (result.error or "")


def test_tool_rejects_unstaged_file_even_under_allowed_root(tmp_path: Path) -> None:
    staged = tmp_path / "staged.csv"
    unstaged = tmp_path / "unstaged.csv"
    staged.write_text("value\n1\n", encoding="utf-8")
    unstaged.write_text("value\n2\n", encoding="utf-8")

    result = registry_for(staged).execute("inspect_file", {"file_path": str(unstaged)})

    assert not result.success
    assert "explicitly staged" in (result.error or "")
