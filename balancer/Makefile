PYTHON ?= python3
VENV := $(CURDIR)/.venv
REPO_ROOT := $(abspath ..)
CONFIG ?= $(CURDIR)/config.json
RUN := cd "$(REPO_ROOT)" && "$(VENV)/bin/python" -m balancer.server --config "$(CONFIG)"

.PHONY: install run install-launchagent print-config

install:
	"$(PYTHON)" -m venv "$(VENV)"
	"$(VENV)/bin/pip" install --upgrade pip
	"$(VENV)/bin/pip" install -r "$(CURDIR)/requirements.txt"

run: install
	$(RUN)

install-launchagent: install
	$(RUN) --install-launchagent

print-config:
	@echo "Config: $(CONFIG)"
