"""Tests for the Studio-only public input boundary."""

from pathlib import Path

import pytest

from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import build_scripted_model
from data_analysis_agent.studio_inputs import (
    StudioInput,
    discover_studio_input_files,
    normalize_studio_input_path,
    prepare_studio_input,
)


def test_studio_schema_exposes_only_question_and_input_data() -> None:
    schema = StudioInput.model_json_schema()

    assert set(schema["properties"]) == {"question", "input_data"}
    assert set(schema["required"]) == {"question", "input_data"}


def test_windows_studio_path_is_mapped_to_wsl_and_unquoted() -> None:
    path = normalize_studio_input_path(' "C:\\Users\\User1\\Downloads\\visits.csv" ')

    assert path == Path("/mnt/c/Users/User1/Downloads/visits.csv")


def test_studio_folder_discovers_supported_files_and_skips_private_inputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "protocol.md").write_text("rules", encoding="utf-8")
    (tmp_path / "data.csv").write_text("value\n10\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    (private / "reference.json").write_text('{"secret": true}', encoding="utf-8")

    files = discover_studio_input_files(tmp_path)

    assert files == [tmp_path / "data.csv", tmp_path / "protocol.md"]


def test_studio_adapter_stages_input_without_changing_normal_graph_inputs(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "measurements.csv"
    data_file.write_text("value\n10\n12\n14\n16\n", encoding="utf-8")
    graph = build_graph(
        build_scripted_model("happy"),
        input_schema=StudioInput,
        input_adapter=prepare_studio_input,
    )

    result = graph.invoke(
        {
            "question": "Calculate the mean and count.",
            "input_data": str(data_file),
        }
    )

    assert result["status"] == "completed"
    assert result["file_paths"] == ["measurements.csv"]
    assert result["input_data"] == str(data_file.resolve())
    assert result["staged_file_paths"] == [str(data_file.resolve())]
    assert "File: measurements.csv" in result["input_context"]


def test_input_schema_and_adapter_are_required_as_a_pair() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        build_graph(build_scripted_model("happy"), input_schema=StudioInput)
