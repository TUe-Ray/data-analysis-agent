"""Compatibility CLI for the importable distributed task builder."""
# ruff: noqa: F403, I001

from data_analysis_agent.task_builders.distributed_longitudinal import *  # noqa: F403


if __name__ == "__main__":
    raise SystemExit(main())  # noqa: F405
