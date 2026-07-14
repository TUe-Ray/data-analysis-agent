.PHONY: install format lint test check api-check

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
