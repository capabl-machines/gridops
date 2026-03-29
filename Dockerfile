FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY gridops/ gridops/
COPY server/ server/
COPY inference.py scripts/ ./

RUN pip install --no-cache-dir .

EXPOSE 8000

ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=10

CMD ["uvicorn", "gridops.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
