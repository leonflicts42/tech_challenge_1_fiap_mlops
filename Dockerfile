FROM python:3.12-slim

WORKDIR /app

# Instala o uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copia dependências
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# Copia APENAS as pastas necessárias
COPY src/ ./src/
COPY tests/ ./tests/

# Expõe a porta
EXPOSE 5000

# O segredo está aqui: chamamos "src.main:app"
# E garantimos que o diretório atual (/app) está no PYTHONPATH
ENV PYTHONPATH=/app
CMD ["/app/.venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]