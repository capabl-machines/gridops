FROM python:3.11-slim

WORKDIR /app

# Install runtime deps first (cached layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    'numpy<2' \
    'pydantic>=2.5' \
    'fastapi' \
    'uvicorn[standard]' \
    'websockets' \
    'openenv-core'

# Copy package + deployment metadata
COPY portfolio_env/ portfolio_env/
COPY openenv.yaml README.md ./
COPY pyproject.toml ./

# Install our package (no-deps because we already pulled deps above)
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=10

CMD ["uvicorn", "portfolio_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
