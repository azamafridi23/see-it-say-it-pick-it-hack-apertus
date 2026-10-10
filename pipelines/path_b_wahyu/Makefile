.DEFAULT_GOAL := test

PIPELINE_DIR := $(shell cd "$$(dirname "$(lastword $(MAKEFILE_LIST))")" && pwd)
ROOT := $(shell cd "$(PIPELINE_DIR)/../.." && pwd)

PYTHON ?= python3
VENV ?= $(ROOT)/.venv
WORKERS ?= 3
DATA ?= $(ROOT)/dataset
OUT ?= $(PIPELINE_DIR)/out/make_test
# A separate cache per run makes the default test use fresh perception.
RUN_ID := $(shell date +%Y%m%d_%H%M%S)_$(shell echo $$$$)
CACHE ?= $(OUT)/cache_$(RUN_ID)

.PHONY: test setup run evaluate unit help

# Recursive targets keep setup, inference, and scoring sequential even with make -j.
test: setup
	$(MAKE) --no-print-directory -C "$(PIPELINE_DIR)" run VENV="$(VENV)" DATA="$(DATA)" OUT="$(OUT)" CACHE="$(CACHE)" WORKERS="$(WORKERS)"
	$(MAKE) --no-print-directory -C "$(PIPELINE_DIR)" evaluate VENV="$(VENV)" DATA="$(DATA)" OUT="$(OUT)"

setup:
	@test -x "$(VENV)/bin/python" || "$(PYTHON)" -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install -r "$(PIPELINE_DIR)/requirements.txt"

run:
	cd "$(ROOT)" && "$(VENV)/bin/python" -m pipelines.path_b_wahyu.pipeline --data "$(DATA)" --workers "$(WORKERS)" --cache-dir "$(CACHE)" --debug-dir "$(OUT)/scenes" --out "$(OUT)/results.json"

evaluate:
	cd "$(ROOT)" && "$(VENV)/bin/python" starter/evaluate.py "$(OUT)/results.json" --data "$(DATA)" --report "$(OUT)/report.json" --robot-targets "$(OUT)/robot_targets.csv"

unit:
	cd "$(ROOT)" && "$(VENV)/bin/python" -m unittest pipelines.path_b_wahyu.test_pipeline -v

help:
	@echo 'make           Install dependencies, run all 10 scenes, then score results.'
	@echo 'make evaluate  Score the existing out/make_test/results.json.'
	@echo 'make unit      Run offline regression checks (after make setup).'
	@echo 'Overrides: WORKERS=1, VENV=/path/to/venv, OUT=/path/to/output, CACHE=/path/to/cache'
	@echo 'Requires API_KEY in the repository-root .env file.'
