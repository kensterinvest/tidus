FROM python:3.12-slim

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (Docker layer cache optimization)
COPY pyproject.toml uv.lock* ./

# Install production dependencies (no dev extras)
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Copy application code
COPY tidus/ ./tidus/
COPY config/ ./config/

# Create data directory for SQLite
RUN mkdir -p /app/data

# Non-root user for security
RUN useradd --create-home --shell /bin/bash tidus && \
    chown -R tidus:tidus /app
USER tidus

EXPOSE 8000

ENV DATABASE_URL="sqlite+aiosqlite:////app/data/tidus.db"
ENV ENVIRONMENT="production"
ENV LOG_LEVEL="INFO"

CMD ["uv", "run", "uvicorn", "tidus.main:app", "--host", "0.0.0.0", "--port", "8000"]
