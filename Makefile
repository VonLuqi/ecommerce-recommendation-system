# =============================================================================
# Makefile — atalhos para tarefas comuns do projeto
# Uso: make <target>
# =============================================================================

.PHONY: help install validate test lint lint-fix \
        pipeline train metrics dvc-init dvc-add-raw \
        docker-up docker-down docker-build docker-ps docker-logs \
        docker-train docker-pipeline docker-from-scratch \
        docker-build-gpu docker-train-gpu docker-pipeline-gpu docker-from-scratch-gpu \
        clean clean-data clean-docker clean-all

# ---------------------------------------------------------------------------
# Variáveis configuráveis
# ---------------------------------------------------------------------------

UV_RUN     ?= uv run
COMPOSE    := docker compose
COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

# =============================================================================
#  Help
# =============================================================================

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║            E-Commerce Recommendation System — Makefile             ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Ambiente:"
	@echo "  make install              Instala dependências (uv sync)"
	@echo "  make validate             Valida o ambiente de desenvolvimento"
	@echo ""
	@echo "Qualidade:"
	@echo "  make test                 Roda a suíte de testes"
	@echo "  make lint                 Roda o linter (ruff)"
	@echo "  make lint-fix             Corrige problemas de lint automaticamente"
	@echo ""
	@echo "Pipeline DVC:"
	@echo "  make pipeline             Executa dvc repro (local)"
	@echo "  make train                Executa treino local (CPU/MPS)"
	@echo "  make metrics              Exibe métricas do último pipeline"
	@echo "  make dvc-init             Inicializa o DVC no repositório"
	@echo "  make dvc-add-raw          Versiona os CSVs brutos com DVC"
	@echo ""
	@echo "Docker — Core:"
	@echo "  make docker-up            Sobe o MLflow Server (porta 5001)"
	@echo "  make docker-down          Derruba todos os serviços"
	@echo "  make docker-build         Builda todas as imagens (CPU — sem dependências de GPU)"
	@echo "  make docker-ps            Status dos containers"
	@echo "  make docker-logs          Logs em tempo real (Ctrl+C para sair)"
	@echo ""
	@echo "Docker — Train & Pipeline (CPU — sem GPU):"
	@echo "  make docker-train         Executa treino no container (CPU)"
	@echo "  make docker-pipeline      Executa dvc repro no container (CPU)"
	@echo "  make docker-from-scratch  Limpa, builda (CPU) e sobe tudo do zero (CPU)"
	@echo ""
	@echo "Docker — GPU (Linux + NVIDIA):"
	@echo "  make docker-build-gpu     Builda todas as imagens com GPU (CUDA)"
	@echo "  make docker-train-gpu     Treino no container com GPU"
	@echo "  make docker-pipeline-gpu  dvc repro no container com GPU"
	@echo "  make docker-from-scratch-gpu Limpa, builda (GPU) e sobe tudo do zero (GPU)"
	@echo ""
	@echo "Limpeza:"
	@echo "  make clean                Remove caches Python"
	@echo "  make clean-data           Remove dados intermediários e modelos"
	@echo "  make clean-docker         Remove containers, volumes e imagens do projeto"
	@echo "  make clean-all            Limpeza completa (clean + clean-data + clean-docker)"
	@echo ""

# =============================================================================
#  Ambiente
# =============================================================================

install:  ## Instala dependências (uv sync)
	uv sync

validate:  ## Valida o ambiente de desenvolvimento
	$(UV_RUN) python scripts/validate_env.py

# =============================================================================
#  Qualidade
# =============================================================================

test:  ## Roda a suíte de testes
	$(UV_RUN) pytest

lint:  ## Roda o linter (ruff)
	$(UV_RUN) ruff check src tests

lint-fix:  ## Corrige problemas de lint automaticamente
	$(UV_RUN) ruff check src tests --fix

# =============================================================================
#  Pipeline DVC
# =============================================================================

dvc-init:  ## Inicializa o DVC no repositório (executar uma vez)
	$(UV_RUN) dvc init

dvc-add-raw:  ## Versiona os CSVs brutos com DVC
	$(UV_RUN) dvc add data/raw/orders.csv
	$(UV_RUN) dvc add data/raw/order_products__prior.csv

pipeline:  ## Executa o pipeline completo via DVC (local)
	$(UV_RUN) dvc repro

train:  ## Executa treino local (CPU/MPS)
	$(UV_RUN) python -m recsys.pipeline.train \
		--input data/processed/train.parquet \
		--output models/model.pkl

metrics:  ## Exibe as métricas do último pipeline executado
	$(UV_RUN) dvc metrics show

# =============================================================================
#  Docker Compose — Core
# =============================================================================

docker-up:  ## Sobe o MLflow Server (porta 5001)
	@echo "🚀 Subindo MLflow Server..."
	$(COMPOSE) up mlflow -d
	@echo ""
	@echo "✅ MLflow disponível em: http://localhost:5001"
	@echo ""

docker-down:  ## Derruba todos os serviços
	@echo "⏹️  Parando serviços..."
	$(COMPOSE) down
	@echo "✅ Serviços parados"

docker-build:  ## Builda todas as imagens (CPU — sem dependências de GPU)
	@echo "🔨 Construindo imagens (CPU)..."
	GPU=false $(COMPOSE) build
	@echo "✅ Build (CPU) concluído"

docker-ps:  ## Status dos containers
	@echo "📋 Status dos containers:"
	@$(COMPOSE) ps

docker-logs:  ## Logs em tempo real (Ctrl+C para sair)
	$(COMPOSE) logs -f

# =============================================================================
#  Docker Compose — Train & Pipeline
# =============================================================================

docker-train:  ## Executa treino no container (CPU)
	@echo "🎓 Executando treino no container..."
	$(COMPOSE) run --rm train
	@echo "✅ Treino concluído"

docker-pipeline:  ## Executa dvc repro no container
	@echo "🔄 Executando pipeline completo (dvc repro) no container..."
	$(COMPOSE) run --rm pipeline
	@echo "✅ Pipeline concluído"

docker-from-scratch:  ## Limpa Docker, builda (CPU) e sobe tudo do zero
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║          🔥 Docker From Scratch (CPU) — Reset                ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(MAKE) clean-docker
	@echo ""
	@$(MAKE) docker-build
	@echo ""
	@$(MAKE) docker-up
	@echo ""
	@echo "✅ Stack CPU pronta! Próximos passos:"
	@echo "   make docker-train       → treina o modelo (CPU)"
	@echo "   make docker-pipeline    → roda pipeline DVC completo (CPU)"
	@echo ""

# =============================================================================
#  Docker Compose — GPU (Linux + NVIDIA)
#
#  ⚠️  Requer:
#    - Linux com drivers NVIDIA
#    - NVIDIA Container Toolkit (nvidia-docker2)
#    - NÃO funciona no macOS (Docker Desktop não suporta GPU passthrough)
# =============================================================================

docker-build-gpu:  ## Builda todas as imagens com GPU (Linux + NVIDIA)
	@echo "🔨 Construindo imagens (GPU)..."
	GPU=true $(COMPOSE) build
	@echo "✅ Build (GPU) concluído"

docker-train-gpu:  ## Treino no container com GPU (Linux + NVIDIA)
	@echo "🎓 Executando treino no container com GPU..."
	$(COMPOSE_GPU) run --rm train
	@echo "✅ Treino (GPU) concluído"

docker-pipeline-gpu:  ## dvc repro no container com GPU (Linux + NVIDIA)
	@echo "🔄 Executando pipeline (dvc repro) no container com GPU..."
	$(COMPOSE_GPU) run --rm pipeline
	@echo "✅ Pipeline (GPU) concluído"

docker-from-scratch-gpu:  ## Limpa Docker, builda (GPU) e sobe tudo do zero
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║          🔥 Docker From Scratch (GPU) — Reset                ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(MAKE) clean-docker
	@echo ""
	@$(MAKE) docker-build-gpu
	@echo ""
	@$(MAKE) docker-up
	@echo ""
	@echo "✅ Stack GPU pronta! Próximos passos:"
	@echo "   make docker-train-gpu    → treina o modelo (GPU)"
	@echo "   make docker-pipeline-gpu → roda pipeline DVC completo (GPU)"
	@echo ""

# =============================================================================
#  Limpeza
# =============================================================================

clean:  ## Remove caches Python (__pycache__, .pytest_cache, .ruff_cache)
	@echo "🧹 Removendo caches Python..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Caches removidos"

clean-data:  ## Remove dados intermediários, modelos e logs gerados
	@echo "🧹 Removendo dados intermediários e modelos..."
	rm -rf data/interim/* data/processed/* models/*
	rm -f metrics.json smoke_test_output.log
	@echo "✅ Dados intermediários removidos"
	@echo "   ℹ️  data/raw/ e mlruns/ foram preservados"

clean-docker:  ## Remove containers, volumes e imagens Docker do projeto
	@echo "🧹 Parando e removendo containers Docker..."
	$(COMPOSE) down -v --remove-orphans
	@echo "🧹 Removendo imagens do projeto..."
	docker images --filter "reference=*ecommerce*" -q | xargs -r docker rmi -f 2>/dev/null || true
	@echo "✅ Recursos Docker do projeto removidos"

clean-all: clean clean-data clean-docker  ## Limpeza completa (caches + dados + Docker)
	@echo ""
	@echo "✅ Limpeza completa finalizada"
	@echo "   ℹ️  Preservados: data/raw/ (Kaggle) e mlruns/ (histórico MLflow)"
