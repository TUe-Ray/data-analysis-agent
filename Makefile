.PHONY: install format lint test check api-check demo-v0-happy demo-v0-replan demo-v0-max-replan demo-v0-live verifier-eval-live verifier-eval-live-3

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	ruff format .

lint:
	ruff check .

test:
	pytest

check:
	ruff format --check .
	ruff check .
	pytest

api-check:
	$(PYTHON) scripts/check_nebius_api.py

demo-v0-happy:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario happy

demo-v0-replan:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario replan

demo-v0-max-replan:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario max-replan

demo-v0-live:
	$(PYTHON) -m data_analysis_agent.demo --mode live \
		--prompt examples/prompts/verifier_trap.txt \
		--file examples/data/measurements_with_missing.csv

verifier-eval-live:
	$(PYTHON) -m data_analysis_agent.demo verifier-eval --repeats 1

verifier-eval-live-3:
	$(PYTHON) -m data_analysis_agent.demo verifier-eval --repeats 3
