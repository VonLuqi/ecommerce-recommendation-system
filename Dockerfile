# =============================================================================
# Stage 1 — builder
# Instala dependências de produção via uv num venv isolado.
# Padrão alinhado com ML_TELCO_CHURN.
# =============================================================================
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=600
ENV UV_CONCURRENT_DOWNLOADS=1

WORKDIR /app

# Copia arquivos de dependências (cache layer)
COPY pyproject.toml uv.lock ./

# Instala dependências sem o projeto (apenas deps de produção)
RUN uv sync --frozen --no-install-project --no-dev


# =============================================================================
# Stage 2 — runtime
# Imagem enxuta de produção: apenas venv + código-fonte.
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Copia ambiente virtual do builder
COPY --from=builder /app/.venv /app/.venv

# Copia código-fonte do projeto
COPY src/ ./src/

# Cria diretórios montados como volumes em runtime
RUN mkdir -p data/raw data/interim data/processed models mlruns

CMD ["python", "-m", "recsys.pipeline.train", \
     "--input",  "data/processed/train.parquet", \
     "--output", "models/model.pkl"]


# =============================================================================
# Stage 3 — pipeline
# Estende runtime com DVC para execução orquestrada via dvc repro.
# Usado pelo serviço `pipeline` no docker-compose.yml.
# =============================================================================
FROM runtime AS pipeline

# Instala DVC, git e pytest (necessários pra rodar dvc repro e testes)
RUN pip install --no-cache-dir "dvc>=3.50" pytest \
    && apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de configuração do pipeline
COPY dvc.yaml dvc.lock params.yaml ./
COPY configs/ ./configs/
COPY .dvc/ ./.dvc/

# Inicializa git repo dentro do container pra DVC funcionar
RUN git init /app \
    && git config --global user.email "app@app" \
    && git config --global user.name "app"

CMD ["dvc", "repro", "--no-commit"]
