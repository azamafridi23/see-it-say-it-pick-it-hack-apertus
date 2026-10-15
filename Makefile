# One command per task. Needs uv (https://docs.astral.sh/uv/); `make help`
# lists the targets. Without make, run the command after each target's
# `uv run`, e.g. uv run python pipelines/path_a/run.py --data scenes

# The folder of scenes for run, run-a and run-b: `make run DATA=my_scenes`.
DATA ?= scenes
NAME = $(notdir $(abspath $(DATA)))
PY = uv run --frozen python

.DEFAULT_GOAL := help
.PHONY: help setup check run run-a run-b heldout-a heldout-b dev-a dev-b \
        test-a test-b app lock clean _scenes

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:  ## Install Python 3.12 and the pinned packages; create .env
	uv sync --frozen
	@test -f .env || { cp .env.example .env; \
	  echo "Created .env: put your key in APERTUS_API_KEY."; }

# heldout/ unless DATA is given: scenes/ is empty in a fresh clone.
check:  ## Check the key, the endpoint and heldout/ (or DATA=...)
	@$(PY) scripts/check_setup.py $(if $(filter command line environment,$(origin DATA)),"$(DATA)")

run: run-a  ## Our submission (Path A) on DATA

run-a: _scenes  ## Path A on DATA -> results/<folder>_path_a.json
	$(PY) pipelines/path_a/run.py --data "$(DATA)" \
	  --out "results/$(NAME)_path_a.json"

run-b: _scenes  ## Path B on DATA -> results/<folder>_path_b.json
	$(PY) pipelines/path_b/run.py --data "$(DATA)" \
	  --out "results/$(NAME)_path_b.json"

heldout-a:  ## Path A on heldout/ -> results_heldout.json
	$(PY) pipelines/path_a/run.py --data heldout

heldout-b:  ## Path B on heldout/ -> results_heldout_path_b.json
	$(PY) pipelines/path_b/run.py --data heldout

dev-a:  ## Path A on the public dataset/, then its score
	$(PY) pipelines/path_a/run.py --data dataset

dev-b:  ## Path B on the public dataset/, then its score
	$(PY) pipelines/path_b/run.py --data dataset

test-a:  ## Path A's offline tests (no key, no downloads)
	$(PY) -m unittest discover -s pipelines/path_a -p 'test_*.py'

test-b:  ## Path B's offline tests (no key, no downloads)
	$(PY) -m unittest discover -s pipelines/path_b/tests

app:  ## The Pick Lab app on http://localhost:3000 (needs Node.js)
	uv sync --frozen --group app
	npm ci
	npm run dev

lock:  ## After changing pyproject.toml: update uv.lock and requirements.txt
	uv lock
	uv export --no-hashes --no-default-groups --no-emit-project --no-header \
	  -q -o requirements.txt
	@printf '%s\n' '# Generated from uv.lock by `make lock`; do not edit. Python 3.12.' \
	  '#   pip install -r requirements.txt' '' | cat - requirements.txt > \
	  requirements.tmp && mv requirements.tmp requirements.txt

clean:  ## Delete the cached scene graphs (the next run rebuilds them)
	rm -rf pipelines/path_a/out pipelines/path_b/out

# Stops early, with a hint, when DATA holds no scenes.
_scenes:
	@ls "$(DATA)"/*/rgb.png >/dev/null 2>&1 || { \
	  echo "No scenes in $(DATA)/: see scenes/README.md for the layout."; \
	  exit 1; }
