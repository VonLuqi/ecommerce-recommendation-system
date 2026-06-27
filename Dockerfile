# =============================================================================
# Stage 1 — builder
# Instala dependências de produção via Poetry num venv isolado.
# O PyTorch é instalado com o índice CPU-only para imagem mais leve.
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# poetry-plugin-export é necessário no Poetry 2.x para o comando `poetry export`
RUN pip install --no-cache-dir "poetry>=2.0" poetry-plugin-export

COPY pyproject.toml poetry.lock ./

# Exporta requirements de produção (sem dev: pytest, ruff, dvc, pre-commit)
RUN poetry export \
        --only main \
        --without-hashes \
        --format requirements.txt \
        --output /tmp/requirements.txt

# Cria venv isolado e instala dependências em 2 passos:
# 1. Todos os pacotes exceto torch e libs NVIDIA/CUDA (que o poetry.lock pinna com GPU)
# 2. torch CPU-only separado — reduz imagem de ~3 GB para ~700 MB
RUN python -m venv /opt/venv \
    && grep -Ev '^(torch|nvidia|triton)' /tmp/requirements.txt > /tmp/requirements_no_torch.txt \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements_no_torch.txt \
    && /opt/venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2"


# =============================================================================
# Stage 2 — runtime
# Imagem enxuta de produção: apenas venv + código-fonte.
# Referência: Docker Aula 03 — multi-stage build e hardening.
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Usuário não-root — boas práticas de segurança (Docker Aula 03)
RUN useradd --no-create-home --shell /bin/false app

# Copia apenas o venv do builder (sem ferramentas de build)
COPY --from=builder /opt/venv /opt/venv

# Copia código-fonte do projeto
COPY src/ ./src/

# Cria diretórios montados como volumes em runtime
RUN mkdir -p data/raw data/interim data/processed models mlruns \
    && chown -R app:app /app

USER app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "recsys.pipeline.train", \
     "--input",  "data/processed/train.parquet", \
     "--output", "models/model.pkl"]


# =============================================================================
# Stage 3 — pipeline
# Estende runtime com DVC para execução orquestrada via dvc repro.
# Usado pelo serviço `pipeline` no docker-compose.yml.
# =============================================================================
FROM runtime AS pipeline

USER root

# Instala DVC, git e pytest (necessários pra rodar dvc repro e testes)
RUN /opt/venv/bin/pip install --no-cache-dir "dvc>=3.50" pytest \
    && apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de configuração do pipeline
COPY dvc.yaml dvc.lock ./
COPY configs/ ./configs/
COPY .dvc/ ./.dvc/

# Inicializa git repo dentro do container pra DVC funcionar
RUN git init /app && git config --global user.email "app@app" && git config --global user.name "app"

RUN chown -R app:app /app

USER app

CMD ["dvc", "repro", "--no-commit"]
