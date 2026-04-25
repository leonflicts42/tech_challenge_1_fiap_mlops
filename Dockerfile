FROM python:3.12-slim

WORKDIR /app

# Instala o uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 1. Copia os arquivos de configuração E os metadados obrigatórios
# O setuptools exige o README para construir o pacote local
COPY pyproject.toml uv.lock* README.md ./

# 2. Copia a pasta src ANTES de sincronizar
# Sem isso, o uv não encontra o diretório 'src' definido no seu TOML
COPY src/ ./src/

# 3. Agora sim, sincroniza as dependências e instala o pacote local
RUN uv sync --frozen --no-dev

# 4. Copia o restante dos arquivos (testes, main, etc)
COPY . .

# Expõe a porta
EXPOSE 5000

# Garante que o ambiente virtual e a pasta src estejam no path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

# Comando para rodar (ajuste o caminho se seu main estiver dentro de churn_telecom)
CMD ["/app/.venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]