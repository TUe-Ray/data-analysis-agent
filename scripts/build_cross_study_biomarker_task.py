"""CLI wrapper for the authoritative static cross-study task builder."""

import sys
from pathlib import Path

from data_analysis_agent.task_builders.cross_study_biomarker_small import main

TASK_ROOT = (
    Path(__file__).resolve().parents[1]
    / "benchmark_tasks"
    / "cross_study_biomarker_harmonization_small"
)


if __name__ == "__main__":
    if "--output" not in sys.argv:
        sys.argv[1:1] = ["--output", str(TASK_ROOT)]
    raise SystemExit(main())
