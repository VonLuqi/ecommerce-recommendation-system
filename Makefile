# =============================================================================
# Makefile — atalhos para tarefas comuns do projeto
# Uso: make <target>
# Windows: requer Git Bash, WSL ou 'make' via winget/chocolatey
# Alternativa Windows nativa: scripts/setup.ps1
# =============================================================================

POETRY := poetry run

.PHONY: help install test lint dvc-init pipeline metrics clean

help:  ## Lista todos os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------

install:  ## Instala todas as dependências (produção + dev)
	poetry install --with dev

# ---------------------------------------------------------------------------
# Qualidade
# ---------------------------------------------------------------------------

test:  ## Roda a suíte de testes
	$(POETRY) pytest

lint:  ## Roda o linter (ruff)
	$(POETRY) ruff check src tests

lint-fix:  ## Corrige problemas de lint automaticamente
	$(POETRY) ruff check src tests --fix

# ---------------------------------------------------------------------------
# Pipeline DVC
# ---------------------------------------------------------------------------

dvc-init:  ## Inicializa o DVC no repositório (executar uma vez)
	$(POETRY) dvc init

dvc-add-raw:  ## Versiona os CSVs brutos com DVC
	$(POETRY) dvc add data/raw/orders.csv
	$(POETRY) dvc add data/raw/order_products__prior.csv

pipeline:  ## Executa o pipeline completo via DVC
	$(POETRY) dvc repro

metrics:  ## Exibe as métricas do último pipeline executado
	$(POETRY) dvc metrics show

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

validate:  ## Valida o ambiente de desenvolvimento
	$(POETRY) python scripts/validate_env.py

clean:  ## Remove caches e artefatos gerados
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
