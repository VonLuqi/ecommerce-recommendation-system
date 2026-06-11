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

## Pré-requisitos

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (para execução via container)
- Conta no [Kaggle](https://www.kaggle.com) (para baixar o dataset)

---

## Setup do Ambiente

### 1. Clonar o repositório

```powershell
git clone <url-do-repositorio>
cd ecommerce-recommendation-system
```

### 2. Instalar dependências

```powershell
poetry install
```

### 3. Configurar variáveis de ambiente

```powershell
copy .env.example .env
```

Edite o `.env` se necessário (os valores padrão funcionam para desenvolvimento local).

### 4. Validar o ambiente

```powershell
poetry run python scripts/validate_env.py
```

---

## Dataset

O projeto usa o **Instacart Market Basket Analysis** (~3M pedidos, ~200K usuários, ~50K produtos).

### Download

1. Acesse: [kaggle.com/competitions/instacart-market-basket-analysis/data](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis?resource=download)
2. Baixe e extraia o arquivo zip
3. Copie **apenas** os dois arquivos necessários:

```
data/
└── raw/
    ├── orders.csv                   (~104 MB)
    └── order_products__prior.csv    (~552 MB)
```

> Os demais arquivos do Kaggle (`products.csv`, `aisles.csv`, etc.) não são utilizados.

---

## Pipeline DVC (local)

O pipeline é orquestrado pelo DVC e executa 4 etapas em sequência:

```
preprocess → feature_eng → train → evaluate
```

### Rodar o pipeline completo

```powershell
poetry run dvc repro
```

O DVC detecta automaticamente o que mudou e só re-executa as etapas necessárias.

### Ver as métricas

```powershell
poetry run dvc metrics show
```

### Publicar os dados no remote DVC

```powershell
poetry run dvc push
```

### Baixar os dados de outro ambiente

```powershell
poetry run dvc pull
```

### Detalhes das etapas

| Etapa | Script | Entrada | Saída |
|---|---|---|---|
| `preprocess` | `recsys/pipeline/preprocess.py` | `data/raw/` | `data/interim/preprocessed.parquet` |
| `feature_eng` | `recsys/pipeline/feature_eng.py` | `preprocessed.parquet` | `train.parquet`, `test.parquet` |
| `train` | `recsys/pipeline/train.py` | `train.parquet` | `models/model.pkl` |
| `evaluate` | `recsys/pipeline/evaluate.py` | `model.pkl` + `test.parquet` + `train.parquet` | `metrics.json` |

### Resultados atuais (baseline de popularidade)

```
hr_at_10        0.038331   → 3,8% das interações de teste têm o item no top-10
coverage_at_10  0.000216   → o modelo cobre 0,02% do catálogo (sempre recomenda os mesmos 10 itens)
num_users       199646
num_interações  2657430
```

> O `Coverage@10` baixo é esperado: o `PopularityRecommender` recomenda os mesmos 10 itens para todos os usuários. O modelo neural (NeuMF) deve superar estes números.

---

## Docker

O projeto tem um `Dockerfile` multi-stage (builder → runtime → pipeline) e um `docker-compose.yml` com 3 serviços.

> **Pré-requisito:** os dados já devem estar em `data/processed/` (rode `dvc repro` antes, ou `dvc pull`).

### 1. Build da imagem

```powershell
docker build --target runtime -t recsys:latest .
```

O PyTorch é instalado com o índice CPU-only, mantendo a imagem em ~500 MB.

### 2. Subir o MLflow Server

```powershell
docker compose up mlflow -d
```

Acesse a UI em `http://localhost:5000`.

### 3. Rodar o treino via container

```powershell
docker compose run --rm train
```

Lê `data/processed/train.parquet` e salva `models/model.pkl` via volume do host.

### 4. Rodar o pipeline completo com DVC dentro do container

```powershell
docker compose run --rm pipeline
```

Executa `dvc repro` dentro do container — integração Docker + DVC.

### 5. Derrubar os serviços

```powershell
docker compose down
```

---

## MLflow

O MLflow é usado para rastrear experimentos, logar métricas e registrar modelos no Model Registry.

### Subir o servidor local

```powershell
docker compose up mlflow -d
```

Acesse `http://localhost:5000` para ver os experimentos, comparar runs e gerenciar modelos.

### Configurar o tracking URI (local sem Docker)

```powershell
# No .env
MLFLOW_TRACKING_URI=mlruns
```

> A integração completa com logging de params, métricas e artefatos será implementada na Etapa 8.

---

## Comandos Úteis

```powershell
# Rodar testes
poetry run pytest

# Lint
poetry run ruff check src tests

# Validar ambiente
poetry run python scripts/validate_env.py

# Testar uma recomendação manualmente
poetry run python -c "
import pickle
model = pickle.load(open('models/model.pkl', 'rb'))
print(model.recommend(user_id='1', top_k=5))
"

# Ver métricas do pipeline
poetry run dvc metrics show

# Verificar status do pipeline DVC
poetry run dvc status
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
├── Dockerfile                      # Multi-stage: builder → runtime → pipeline
├── docker-compose.yml              # Serviços: mlflow, train, pipeline
├── Makefile                        # Atalhos para tarefas comuns
├── dvc.yaml                        # Pipeline DVC (4 etapas)
├── dvc.lock                        # Hashes dos artefatos (gerado por dvc repro)
├── metrics.json                    # Métricas do último evaluate
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