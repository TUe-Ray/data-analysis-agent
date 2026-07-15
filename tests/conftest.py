"""Global test isolation for filesystem artifacts."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default graph run paths inside pytest's per-test temporary area."""
    monkeypatch.chdir(tmp_path)
