FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 nad

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

RUN mkdir -p /app/artifacts /app/data/raw /app/data/processed \
    && chown -R nad:nad /app

USER nad

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "nad_similarity.api:app", "--host", "0.0.0.0", "--port", "8000"]
