FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY examples ./examples
COPY assets ./assets

RUN pip install --upgrade pip setuptools wheel ; \
    pip install .[all]

RUN mkdir -p /app/outputs

# Default container behavior: show the built-in demo help.
# Override this at runtime, for example:
#   docker run --rm geoprompt geoprompt-demo
#   docker run --rm geoprompt python -m pytest
CMD ["geoprompt-demo", "--help"]
