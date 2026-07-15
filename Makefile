.PHONY: install format lint test check api-check demo-v0-happy demo-v0-replan demo-v0-max-replan demo-v0-valid-json demo-v0-output-repair demo-v0-output-failure demo-v0-live demo-tools-success demo-python-success demo-python-repair demo-python-failure verifier-eval-live verifier-eval-live-3

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
RUFF = $(PYTHON) -m ruff
PYTEST = $(PYTHON) -m pytest

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(RUFF) format .

lint:
	$(RUFF) check .

test:
	$(PYTEST)

check:
	$(RUFF) format --check .
	$(RUFF) check .
	$(PYTEST)

api-check:
	$(PYTHON) scripts/check_nebius_api.py

demo-v0-happy:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario happy

demo-v0-replan:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario replan

demo-v0-max-replan:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario max-replan

demo-v0-valid-json:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario valid-json

demo-v0-output-repair:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario output-repair

demo-v0-output-failure:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario output-failure

demo-v0-live:
	$(PYTHON) -m data_analysis_agent.demo --mode live \
		--prompt examples/prompts/verifier_trap.txt \
		--file examples/data/measurements_with_missing.csv

demo-tools-success:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario trusted-tools-success

demo-python-success:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario generated-python-success

demo-python-repair:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario generated-python-repair

demo-python-failure:
	$(PYTHON) -m data_analysis_agent.demo --mode offline --scenario generated-python-failure

verifier-eval-live:
	$(PYTHON) -m data_analysis_agent.demo verifier-eval --repeats 1

verifier-eval-live-3:
	$(PYTHON) -m data_analysis_agent.demo verifier-eval --repeats 3
