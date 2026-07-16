"""Regression coverage for the static orchestration-diagram exporter."""

import importlib.util
from pathlib import Path


def _load_export_module():
    script_path = Path(__file__).parents[1] / "scripts" / "export_graph.py"
    spec = importlib.util.spec_from_file_location("export_graph_script", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Drawable:
    def draw_mermaid(self) -> str:
        return "graph TD; START --> planner"

    def draw_mermaid_png(self) -> bytes:
        return b"png-bytes"


class _Graph:
    def get_graph(self) -> _Drawable:
        return _Drawable()


def test_export_graph_writes_mermaid_and_png(tmp_path: Path) -> None:
    exporter = _load_export_module()

    exporter.export_graph(_Graph(), tmp_path)

    assert (
        tmp_path / "agent_workflow.mmd"
    ).read_text() == "graph TD; START --> planner"
    assert (tmp_path / "agent_workflow.png").read_bytes() == b"png-bytes"


def test_export_graph_can_skip_png(tmp_path: Path) -> None:
    exporter = _load_export_module()

    exporter.export_graph(_Graph(), tmp_path, include_png=False)

    assert (tmp_path / "agent_workflow.mmd").is_file()
    assert not (tmp_path / "agent_workflow.png").exists()
