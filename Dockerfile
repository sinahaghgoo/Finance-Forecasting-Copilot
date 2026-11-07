FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY fin_copilot ./fin_copilot
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./

RUN pip install --upgrade pip \
    && pip install .

RUN mkdir -p data/raw data/processed artifacts

ENTRYPOINT ["fincopilot"]
CMD ["--help"]
