# syntax=docker/dockerfile:1
#
# Imagen de NFL-Predictor.
#
# NOTA DE HONESTIDAD: este Dockerfile se entrega según lo pedido, pero NO ha podido
# verificarse: la máquina de desarrollo no tiene Docker instalado. La sintaxis es
# correcta y sigue las prácticas habituales, pero no se ha construido ni ejecutado.
# Todo lo demás del proyecto (DuckDB, MLflow, Prefect, Streamlit) sí corre y se
# prueba localmente sin contenedores.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    GIT_PYTHON_REFRESH=quiet

# libgomp1 es necesaria para LightGBM y XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Capa de dependencias separada: cambiar el código no invalida la instalación.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --no-dev || uv sync --no-install-project --no-dev

COPY src/ ./src/
COPY migrations/ ./migrations/
COPY config/ ./config/
COPY README.md ./

RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Usuario sin privilegios.
RUN useradd --create-home --uid 1000 nflp \
    && mkdir -p /app/data \
    && chown -R nflp:nflp /app
USER nflp

VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "src/nflp/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
