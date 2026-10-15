# `make run` is what the judges run: it builds the Docker image and runs our
# submission (Path A) in it. `make help` lists every target. The other
# targets run without Docker and need uv (https://docs.astral.sh/uv/).

# The folder of scenes: `make run DATA=path/to/scenes`. Defaults to the
# five held-out scenes.
DATA ?= data/heldout
NAME = $(notdir $(abspath $(DATA)))
# heldout gives results/results_heldout.json; any other folder
# results/<folder>_path_a.json.
OUT = $(if $(filter heldout,$(NAME)),results_heldout.json,$(NAME)_path_a.json)
IMAGE ?= asian-prompters-path-a
PY = uv run --frozen python

.DEFAULT_GOAL := help
.PHONY: help run build setup check run-a run-b heldout-a heldout-b dev-a \
        dev-b test-a test-b app lock clean _scenes

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

run: build _scenes  ## Path A in Docker on DATA -> results/
	@mkdir -p results
	docker run --rm \
	  $(if $(wildcard .env),-v "$(CURDIR)/.env:/app/.env:ro") \
	  -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY \
	  -v "$(abspath $(DATA)):/scenes:ro" -v "$(CURDIR)/results:/out" \
	  $(IMAGE) --data /scenes --out "/out/$(OUT)"

build:  ## Build the Docker image (CPU-only packages and the model weights)
	docker build -t $(IMAGE) .

setup:  ## Without Docker: install Python 3.12 and the packages; create .env
	uv sync --frozen
	@test -f .env || { cp .env.example .env; \
	  echo "Created .env: put your key in LLM_API_KEY."; }

check:  ## Without Docker: check the key, the endpoint and DATA
	@$(PY) src/scripts/check_setup.py "$(DATA)"

run-a: _scenes  ## Without Docker: Path A on DATA -> results/
	$(PY) src/pipelines/path_a/run.py --data "$(DATA)" --out "results/$(OUT)"

run-b: _scenes  ## Without Docker: Path B on DATA -> results/
	$(PY) src/pipelines/path_b/run.py --data "$(DATA)" \
	  --out "results/$(NAME)_path_b.json"

heldout-a:  ## Without Docker: Path A on data/heldout -> results_heldout.json
	$(PY) src/pipelines/path_a/run.py --data data/heldout

heldout-b:  ## Without Docker: Path B on data/heldout
	$(PY) src/pipelines/path_b/run.py --data data/heldout

dev-a:  ## Without Docker: Path A on the public data/dataset, then its score
	$(PY) src/pipelines/path_a/run.py --data data/dataset

dev-b:  ## Without Docker: Path B on the public data/dataset, then its score
	$(PY) src/pipelines/path_b/run.py --data data/dataset

test-a:  ## Path A's offline tests (no key, no downloads)
	$(PY) -m unittest discover -s src/pipelines/path_a -p 'test_*.py'

test-b:  ## Path B's offline tests (no key, no downloads)
	$(PY) -m unittest discover -s src/pipelines/path_b/tests

app:  ## The Pick Lab app on http://localhost:3000 (needs Node.js)
	uv sync --frozen --group app
	cd src && npm ci && npm run dev

lock:  ## After changing pyproject.toml: update uv.lock and requirements.txt
	uv lock
	uv export --no-hashes --no-default-groups --no-emit-project --no-header \
	  -q -o requirements.txt
	@printf '%s\n' '# Generated from uv.lock by `make lock`; do not edit. Python 3.12.' \
	  '#   pip install -r requirements.txt' \
	  '# Linux takes the CPU-only torch from PyTorch'"'"'s own index:' \
	  '--extra-index-url https://download.pytorch.org/whl/cpu' '' \
	  | cat - requirements.txt > \
	  requirements.tmp && mv requirements.tmp requirements.txt

clean:  ## Delete the cached scene graphs (the next run rebuilds them)
	rm -rf src/pipelines/path_a/out src/pipelines/path_b/out

# Stops early, with a hint, when DATA holds no scenes.
_scenes:
	@ls "$(DATA)"/*/rgb.png >/dev/null 2>&1 || ls "$(DATA)"/rgb.png \
	  >/dev/null 2>&1 || { \
	  echo "No scenes in $(DATA)/: see data/scenes/README.md for the layout."; \
	  exit 1; }
