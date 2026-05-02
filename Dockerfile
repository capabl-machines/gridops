FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir numpy pydantic fastapi "uvicorn[standard]" websockets openai requests openenv-core

# Copy app code
COPY gridops/ gridops/
COPY server/ server/
COPY assets/ assets/
COPY evals/ evals/
COPY inference.py openenv.yaml README.md ./
COPY scripts/ scripts/

EXPOSE 8000

ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=10

CMD ["uvicorn", "gridops.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
