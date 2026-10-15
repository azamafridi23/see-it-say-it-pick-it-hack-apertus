# Path A (our submission) in a container: `make run` builds and runs it.
FROM python:3.12-slim

# OpenCV needs these system libraries on a slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH"

# The pinned packages first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# The open-weight models, baked in: a run needs no 3.4 GB download.
COPY src/pipelines/path_a/workcell.py src/pipelines/path_a/workcell.ini \
     src/pipelines/path_a/
COPY src/scripts/download_models.py src/scripts/
RUN python src/scripts/download_models.py

COPY . .

# The scenes are mounted at /scenes and the answers written to /out.
ENTRYPOINT ["python", "src/pipelines/path_a/run.py"]
CMD ["--data", "/scenes", "--out", "/out/results_heldout.json"]
