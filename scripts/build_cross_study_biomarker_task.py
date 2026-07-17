"""CLI wrapper for the importable cross-study task builder."""
# ruff: noqa: I001

from data_analysis_agent.task_builders.cross_study_biomarker import main


if __name__ == "__main__":
    raise SystemExit(main())
