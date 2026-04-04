PYTHON ?= python3
VENV := $(CURDIR)/.venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: install-dev test coverage clean

$(VENV)/bin/python:
	"$(PYTHON)" -m venv "$(VENV)"
	"$(PIP)" install --upgrade pip
	"$(PIP)" install -r requirements-dev.txt

install-dev: $(VENV)/bin/python

test: install-dev
	PYTHONPATH="$(CURDIR)" "$(PYTEST)" tests

coverage: install-dev
	PYTHONPATH="$(CURDIR)" "$(PYTEST)" tests --cov-report=html

clean:
	rm -rf "$(VENV)" .pytest_cache htmlcov .coverage
