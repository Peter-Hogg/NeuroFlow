FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/neuroflow
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY neuroflow neuroflow
COPY neuroflow_cellpose neuroflow_cellpose
COPY neuroflow_pynapple neuroflow_pynapple
COPY examples examples
RUN uv sync --locked --no-dev
ENTRYPOINT ["uv", "run", "neuroflow"]
