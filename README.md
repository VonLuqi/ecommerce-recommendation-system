# E-Commerce Recommendation System

![Fase](https://img.shields.io/badge/fase-pipeline-blue)
![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20MLflow%20%7C%20DVC%20%7C%20Docker-informational)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Testes](https://img.shields.io/badge/testes-33%20passando-brightgreen)

Sistema de recomendação de produtos para e-commerce baseado no histórico de compras do usuário, implementado com redes neurais do tipo embedding-based (NeuMF). Dataset: **Instacart Market Basket**.

---

## Objetivo

Dado o histórico de compras de um usuário, ranquear produtos com maior probabilidade de serem comprados novamente, aumentando a taxa de conversão e a relevância das recomendações.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Modelagem | PyTorch, Scikit-Learn |
| Rastreamento de experimentos | MLflow |
| Versionamento de dados | DVC |
| Empacotamento | Docker |
| Dependências | Poetry |
| Linguagem | Python 3.11+ |

---

## Instalação

### Windows (PowerShell)

```powershell
git clone <url-do-repositorio>
cd ecommerce-recommendation-system
.\scripts\setup.ps1
```

O script instala o Poetry (se necessário), configura o PATH, instala todas as dependências e cria o `.env`.

### Linux / macOS

```bash
git clone <url-do-repositorio>
cd ecommerce-recommendation-system
pip install poetry
make install
```

---

## Dataset

O projeto usa o **Instacart Market Basket Analysis** (~3M pedidos, ~200K usuários, ~50K produtos).

1. Acesse: [kaggle.com/competitions/instacart-market-basket-analysis/data](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
2. Baixe e extraia o arquivo zip
3. Copie os dois arquivos necessários:

```
data/
└── raw/
    ├── orders.csv
    └── order_products__prior.csv
```

> Os demais arquivos do Kaggle (`products.csv`, `aisles.csv`, etc.) não são necessários neste momento.

---

## Executando o Pipeline

O pipeline é orquestrado pelo DVC e executa 4 etapas em sequência:

```
preprocess → feature_eng → train → evaluate
```

```powershell
# 1. Inicializar o DVC (apenas na primeira vez)
poetry run dvc init

# 2. Versionar os dados brutos
poetry run dvc add data/raw/orders.csv
poetry run dvc add data/raw/order_products__prior.csv

# 3. Rodar o pipeline completo
poetry run dvc repro

# 4. Ver as métricas
poetry run dvc metrics show
```

| Etapa | Script | Entrada | Saída |
|---|---|---|---|
| `preprocess` | `recsys/pipeline/preprocess.py` | `data/raw/` | `data/interim/preprocessed.parquet` |
| `feature_eng` | `recsys/pipeline/feature_eng.py` | `preprocessed.parquet` | `train.parquet`, `test.parquet` |
| `train` | `recsys/pipeline/train.py` | `train.parquet` | `models/model.pkl` |
| `evaluate` | `recsys/pipeline/evaluate.py` | `model.pkl` + `test.parquet` | `metrics.json` |

---

## Comandos Úteis

```powershell
# Rodar testes
poetry run pytest

# Lint
poetry run ruff check src tests

# Validar ambiente
poetry run python scripts/validate_env.py
```

Com `make` (Linux/macOS/Git Bash):

```bash
make install     # instala dependências
make test        # roda testes
make lint        # roda o linter
make pipeline    # executa dvc repro
make metrics     # exibe métricas
make help        # lista todos os comandos
```

---

## Estrutura do Repositório

```
ecommerce-recommendation-system/
├── Makefile                        # Atalhos para tarefas comuns
├── dvc.yaml                        # Pipeline DVC (4 etapas)
├── pyproject.toml                  # Dependências e configuração
├── ruff.toml                       # Configuração do linter
├── .env.example                    # Variáveis de ambiente (copie para .env)
├── data/
│   ├── raw/                        # CSVs do Instacart (gerenciados pelo DVC)
│   ├── interim/                    # Dados pré-processados
│   └── processed/                  # Features prontas para treino
├── models/                         # Modelos serializados (.pkl)
├── scripts/
│   ├── setup.ps1                   # Setup automático para Windows
│   └── validate_env.py             # Valida o ambiente de desenvolvimento
├── src/recsys/
│   ├── config/settings.py          # Configuração central (Pydantic Settings)
│   ├── data/
│   │   ├── loader.py               # Interface abstrata BaseInteractionLoader
│   │   └── instacart_loader.py     # Carregador concreto do Instacart
│   ├── pipeline/
│   │   ├── preprocess.py           # Etapa 1: limpeza + k-core filtering
│   │   ├── feature_eng.py          # Etapa 2: encode IDs + split treino/teste
│   │   ├── train.py                # Etapa 3: treino do modelo baseline
│   │   └── evaluate.py             # Etapa 4: HR@K e Coverage@K
│   ├── recommenders/
│   │   ├── base.py                 # Interface Strategy BaseRecommender
│   │   └── popularity.py           # Baseline: popularidade global
│   └── utils/seeds.py              # Fixação de seeds globais
├── tests/
│   ├── test_data_pipeline.py       # Testes do loader e funções de pipeline
│   ├── test_recommenders.py        # Testes dos recomendadores
│   └── test_settings.py            # Testes de configuração
└── configs/default.yaml            # Parâmetros padrão do projeto
```

---

## Convenções de Desenvolvimento

### Branches

```
main                              → código estável/entregável
develop                           → integração contínua da equipe
feature/<escopo>/<descricao>      → ex: feature/data/instacart-loader
fix/<descricao>                   → ex: fix/embedding-shape
docs/<descricao>                  → ex: docs/model-card
model/<descricao>                 → ex: model/neumf-architecture
```

### Commits Semânticos

```
feat:      nova funcionalidade
fix:       correção de bug
docs:      documentação
data:      pipeline ou transformação de dados
model:     arquitetura, treino ou avaliação
refactor:  refatoração sem mudança de comportamento
test:      testes
chore:     build, dependências, configuração
```

---

## Referências

- He, X. et al. (2017). *Neural Collaborative Filtering*. WWW '17.
- Instacart. *Instacart Market Basket Analysis*. Kaggle, 2017.
