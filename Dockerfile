ARG PYTHON_IMAGE=python:3.10.19-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG UV_VERSION=0.10.4
ARG NEUROFLOW_GIT_SHA=unknown
ARG NEUROFLOW_GIT_DIRTY=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/neuroflow/.venv \
    NEUROFLOW_GIT_SHA=${NEUROFLOW_GIT_SHA} \
    NEUROFLOW_GIT_DIRTY=${NEUROFLOW_GIT_DIRTY}
WORKDIR /opt/neuroflow
RUN pip install --no-cache-dir "uv==${UV_VERSION}"
COPY pyproject.toml uv.lock README.md .python-version ./
COPY neuroflow neuroflow
COPY neuroflow_cellpose neuroflow_cellpose
COPY neuroflow_pynapple neuroflow_pynapple
COPY examples examples
RUN uv sync --locked --no-dev --no-editable
ENTRYPOINT ["/opt/neuroflow/.venv/bin/neuroflow"]
