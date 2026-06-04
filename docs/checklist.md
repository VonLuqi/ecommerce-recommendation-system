# Checklist de Progresso — E-Commerce Recommendation System

> Legenda de status:  
> ✅ Decidido/Concluído | 🔲 Pendente | 🧪 Depende de validação experimental  
>
> A estrutura segue a ordem cronológica do [Tech Challenge](../tech-challenge-ordenacao.md).  
> O responsável por cada bloco está indicado no cabeçalho da seção.

---

## Bloco 1 — Kickoff e Definição do Problema
**Responsável: Eu** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 1.1 | Escolher o dataset de interações user-item | ✅ | Amazon Reviews 2023 "Toys and Games" |
| 1.2 | Validar se o dataset tem volume suficiente | ✅ | ~165K interações — 16× o mínimo de 10K |
| 1.3 | Definir o problema de negócio do e-commerce | ✅ | Ver [kickoff.md §2](kickoff.md#2-definição-do-problema-de-negócio) |
| 1.4 | Definir o objetivo do sistema de recomendação | ✅ | Top-K ranking; métrica principal NDCG@10 |
| 1.5 | Escolher abordagem principal: MLP ou embedding-based | ✅ | NeuMF (GMF + MLP) — ver [decisions.md ADR-002](decisions.md#adr-002) |
| 1.6 | Definir integrantes e responsabilidades | ✅ | 4 integrantes mapeados aos 10 blocos do plano |
| 1.7 | Criar o repositório GitHub | ✅ | Repositório inicializado |
| 1.8 | Definir padrão de branch, PR e commits semânticos | ✅ | Git Flow simplificado + Conventional Commits |

---

## Bloco 2 — Estrutura Base do Projeto
**Responsável: Eu** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 2.1 | Criar `src/`, `tests/`, `data/`, `models/`, `configs/` | ✅ | Estrutura criada |
| 2.2 | Configurar `.gitignore` | ✅ | Cobre Python, dados, MLflow, IDEs |
| 2.3 | Configurar `.dockerignore` | ✅ | Exclui dados, cache e segredos |
| 2.4 | Configurar `.env.example` | ✅ | Documenta todas as variáveis do projeto |
| 2.5 | Aplicar naming conventions | ✅ | snake_case em módulos; PascalCase em classes |
| 2.6 | Aplicar princípios SOLID | ✅ | SRP, OCP, LSP, ISP, DIP em `base.py` e `popularity.py` |
| 2.7 | Manter módulos curtos e funções pequenas | ✅ | Nenhum módulo > 70 linhas |
| 2.8 | Implementar pelo menos 1 design pattern | ✅ | **Strategy Pattern** — `BaseRecommender` + `PopularityRecommender` |
| 2.9 | Adicionar type hints nas funções públicas | ✅ | 100% das funções públicas anotadas |
| 2.10 | Padronizar docstrings em estilo Google | ✅ | Args, Returns, Raises, Example em todos os módulos |
| 2.11 | Configurar `ruff` | ✅ | `ruff.toml` com regras E/F/I/N/UP/ANN/S/B/C4/SIM |
| 2.12 | Configurar `pre-commit` | ✅ | `.pre-commit-config.yaml` com ruff + hooks utilitários |

---

## Bloco 3 — Ambiente e Dependências
**Responsável: Eu** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 3.1 | Configurar `pyproject.toml` com Poetry ou uv | ✅ | Poetry 2.4.1 — grupos main + dev |
| 3.2 | Separar dependências de produção e desenvolvimento | ✅ | `[project.dependencies]` + `[tool.poetry.group.dev.dependencies]` |
| 3.3 | Adicionar PyTorch | ✅ | `torch>=2.2` |
| 3.4 | Adicionar Scikit-Learn | ✅ | `scikit-learn>=1.4` |
| 3.5 | Adicionar MLflow | ✅ | `mlflow>=2.12` |
| 3.6 | Adicionar pytest | ✅ | `pytest>=8.0` + `pytest-cov>=5.0` (dev) |
| 3.7 | Adicionar ruff | ✅ | `ruff>=0.4` (dev) |
| 3.8 | Gerar e commitar lock file | ✅ | `poetry.lock` gerado com Python 3.12 |
| 3.9 | Configurar variáveis em `.env` | ✅ | `.env` criado (não commitado); `.env.example` atualizado |
| 3.10 | Configurar Pydantic Settings | ✅ | `src/recsys/config/settings.py` — AppSettings + sub-modelos |
| 3.11 | Criar `scripts/validate_env.py` | ✅ | 4/4 verificações passam (Python, deps, env vars, dirs) |
| 3.12 | Validar instalação limpa em ambiente novo | ✅ | `poetry install` + `validate_env.py` — 4/4 ✓ |
| 3.13 | Fixar seeds globais | ✅ | `src/recsys/utils/seeds.py` — random, numpy, torch |

---

## Bloco 4 — Dados e Preparação do Pipeline
**Responsável: Pedro** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 4.1 | Baixar e organizar o dataset bruto | 🔲 | `McAuley-Lab/Amazon-Reviews-2023` via HuggingFace |
| 4.2 | Definir estratégia de ingestão dos dados | 🔲 | |
| 4.3 | Definir fluxo `preprocess → feature_eng → train → evaluate` | 🔲 | |
| 4.4 | Implementar preprocessamento | 🔲 | Inclui filtro k-core |
| 4.5 | Implementar feature engineering | 🔲 | |
| 4.6 | **Validar threshold de rating por EDA** | 🧪 | Proposta: ≥ 4 — confirmar com histograma da distribuição real (ver ADR-003) |
| 4.7 | Definir artefatos gerados por cada etapa | 🔲 | |
| 4.8 | Inicializar DVC | 🔲 | |
| 4.9 | Versionar dataset com DVC | 🔲 | |
| 4.10 | Configurar remote do DVC | 🔲 | |
| 4.11 | Criar `dvc.yaml` | 🔲 | |
| 4.12 | Validar execução com `dvc repro` | 🔲 | |

---

## Bloco 5 — Containerização e Execução Reprodutível
**Responsável: Pedro** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 5.1 | Criar `Dockerfile` multi-stage | 🔲 | |
| 5.2 | Otimizar a imagem | 🔲 | |
| 5.3 | Criar `docker-compose.yml` | 🔲 | |
| 5.4 | Subir serviço de treino | 🔲 | |
| 5.5 | Subir MLflow Server | 🔲 | |
| 5.6 | Testar execução do projeto via container | 🔲 | |
| 5.7 | Validar integração entre Docker e DVC | 🔲 | |

---

## Bloco 6 — Baselines e Critérios de Avaliação
**Responsável: Victor** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 6.1 | Implementar baselines com Scikit-Learn | 🔲 | Ex: Most Popular, Random, User-Based CF |
| 6.2 | Definir 4 métricas de avaliação | 🔲 | NDCG@10 (principal), HR@10, Precision@10, Recall@10 |
| 6.3 | Rodar avaliação inicial dos baselines | 🔲 | |
| 6.4 | Salvar resultados comparáveis | 🔲 | |
| 6.5 | Garantir que etapa `evaluate` gera métricas confiáveis | 🔲 | |

---

## Bloco 7 — Modelo Neural
**Responsável: Victor** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 7.1 | Implementar o NeuMF em PyTorch | 🔲 | Ramos GMF + MLP (ver ADR-002) |
| 7.2 | Integrar o modelo ao pipeline DVC | 🔲 | |
| 7.3 | Configurar treino reproduzível (seeds fixas) | 🔲 | |
| 7.4 | Adicionar early stopping | 🔲 | |
| 7.5 | Rodar experimentos iniciais | 🔲 | |
| 7.6 | Ajustar hiperparâmetros | 🔲 | Embedding dim, camadas MLP, dropout, LR |
| 7.7 | Comparar NeuMF vs baselines (NDCG@10 / HR@10) | 🔲 | |

---

## Bloco 8 — Tracking e Registro do Modelo
**Responsável: Eduardo** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 8.1 | Configurar MLflow tracking | 🔲 | |
| 8.2 | Logar parâmetros | 🔲 | |
| 8.3 | Logar métricas | 🔲 | Incluir NDCG@10 e HR@10 em todos os runs |
| 8.4 | Logar artefatos | 🔲 | |
| 8.5 | Garantir ao menos 3 runs rastreados | 🔲 | |
| 8.6 | Registrar o melhor modelo no Model Registry | 🔲 | |
| 8.7 | Promover modelo para Staging | 🔲 | |
| 8.8 | Promover modelo para Production | 🔲 | |

---

## Bloco 9 — Documentação Técnica
**Responsável: Eduardo** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 9.1 | Escrever o Model Card | 🔲 | |
| 9.2 | Documentar performance do modelo | 🔲 | |
| 9.3 | Documentar limitações | 🔲 | |
| 9.4 | Documentar vieses | 🔲 | |
| 9.5 | Finalizar README com instruções de setup | 🔲 | |
| 9.6 | Incluir instruções de treino | 🔲 | |
| 9.7 | Incluir instruções de DVC | 🔲 | |
| 9.8 | Incluir instruções de Docker | 🔲 | |
| 9.9 | Incluir instruções de MLflow | 🔲 | |

---

## Bloco 10 — Validação Final
**Responsável: Eduardo** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 10.1 | Rodar lint final (`ruff`) | 🔲 | |
| 10.2 | Rodar testes finais (`pytest`) | 🔲 | |
| 10.3 | Validar instalação limpa | 🔲 | |
| 10.4 | Validar `dvc repro` | 🔲 | |
| 10.5 | Validar Docker Compose | 🔲 | |
| 10.6 | Validar MLflow e Model Registry | 🔲 | |
| 10.7 | Revisar estrutura do repositório | 🔲 | |
| 10.8 | Revisar histórico de commits | 🔲 | |
| 10.9 | Conferir todos os critérios de avaliação | 🔲 | |

---

## Resumo de Progresso por Bloco

| Bloco | Responsável | Decididos/Concluídos | Pendentes | Aguarda experimento |
|---|---|---|---|---|
| 1 — Kickoff | Eu | 8 | 0 | 0 |
| 2 — Estrutura base | Eu | 12 | 0 | 0 |
| 3 — Ambiente | Eu | 13 | 0 | 0 |
| 4 — Dados | Pedro | 0 | 11 | 1 |
| 5 — Containerização | Pedro | 0 | 7 | 0 |
| 6 — Baselines | Victor | 0 | 5 | 0 |
| 7 — Modelo neural | Victor | 0 | 7 | 0 |
| 8 — Tracking | Eduardo | 0 | 8 | 0 |
| 9 — Documentação | Eduardo | 0 | 9 | 0 |
| 10 — Validação final | Eduardo | 0 | 9 | 0 |
| **Total** | | **8** | **81** | **1** |


---

## Fase 1 — Kickoff e Definição do Problema

| # | Item | Status | Observações |
|---|---|---|---|
| 1.1 | Definição do problema de negócio | ✅ | Ver [kickoff.md §2](kickoff.md#2-definição-do-problema-de-negócio) |
| 1.2 | Definição do objetivo do recomendador | ✅ | Top-K ranking; métrica principal HR@10 |
| 1.3 | Comparação de 3 datasets públicos | ✅ | Amazon, MovieLens 1M, H&M Fashion |
| 1.4 | Escolha do dataset principal | ✅ | Amazon Reviews 2023 "Toys and Games" |
| 1.5 | Validação do volume mínimo (≥ 10K interações) | ✅ | ~165K avaliações (16× o mínimo) |
| 1.6 | Decisão de abordagem do modelo | ✅ | NeuMF (GMF + MLP) — ver [decisions.md ADR-002](decisions.md#adr-002) |
| 1.7 | Divisão inicial de responsabilidades | ✅ | Ver [kickoff.md §7](kickoff.md#7-divisão-de-responsabilidades) |
| 1.8 | Convenções de branch / PR / commits | ✅ | Git Flow simplificado + Conventional Commits |
| 1.9 | Criação do README inicial | ✅ | [README.md](../README.md) |
| 1.10 | Criação do docs/kickoff.md | ✅ | [kickoff.md](kickoff.md) |
| 1.11 | Criação do docs/decisions.md | ✅ | [decisions.md](decisions.md) |
| 1.12 | Confirmação do dataset pela equipe | 🟡 | Subconjunto "Toys and Games" ou outro domínio Amazon? |
| 1.13 | Definição do threshold de rating (feedback implícito) | 🟡 | Proposta atual: ≥ 4 — confirmar após EDA |
| 1.14 | Escolha da métrica principal de avaliação | 🟡 | HR@10 ou NDCG@10 como métrica reportada? |
| 1.15 | Alinhamento do tamanho da equipe com os papéis | 🟡 | Ajustar papéis ao número real de integrantes |

---

## Fase 2 — Engenharia de Dados

| # | Item | Status | Observações |
|---|---|---|---|
| 2.1 | Configuração do ambiente virtual Python | 🔲 | |
| 2.2 | Download e inspeção inicial do dataset | 🔲 | |
| 2.3 | Análise exploratória de dados (EDA) | 🔲 | Distribuição de ratings, esparsidade, usuários/itens |
| 2.4 | Limpeza e filtro k-core | 🔲 | Remover usuários/itens com < k interações (k=5 sugerido) |
| 2.5 | Divisão treino/validação/teste | 🔲 | Leave-one-out ou temporal split |
| 2.6 | Geração de negative samples para treino | 🔲 | |
| 2.7 | Configuração do DVC para versionamento de dados | 🔲 | |
| 2.8 | Notebook de EDA finalizado e revisado | 🔲 | |

---

## Fase 3 — Modelagem

| # | Item | Status | Observações |
|---|---|---|---|
| 3.1 | Implementação do NeuMF em PyTorch | 🔲 | Ramos GMF e MLP |
| 3.2 | Dataset class e DataLoader do PyTorch | 🔲 | |
| 3.3 | Loop de treino com BCE Loss | 🔲 | |
| 3.4 | Avaliação HR@10 e NDCG@10 | 🔲 | |
| 3.5 | Baseline de popularidade implementado | 🔲 | |
| 3.6 | Comparação NeuMF vs baseline | 🔲 | |
| 3.7 | Tuning de hiperparâmetros | 🔲 | Embedding dim, camadas MLP, dropout, LR |

---

## Fase 4 — MLOps

| # | Item | Status | Observações |
|---|---|---|---|
| 4.1 | Configuração do MLflow (tracking server local) | 🔲 | |
| 4.2 | Logging de métricas e parâmetros no MLflow | 🔲 | |
| 4.3 | Registro do modelo no MLflow Model Registry | 🔲 | |
| 4.4 | Criação do Dockerfile | 🔲 | |
| 4.5 | Docker Compose para ambiente reproduzível | 🔲 | |
| 4.6 | Pipeline DVC end-to-end | 🔲 | |
| 4.7 | CI básico (GitHub Actions) | 🔲 | |

---

## Fase 5 — Documentação Final e Apresentação

| # | Item | Status | Observações |
|---|---|---|---|
| 5.1 | Relatório técnico final | 🔲 | |
| 5.2 | Slides de apresentação | 🔲 | |
| 5.3 | README final com instruções de execução | 🔲 | |
| 5.4 | Checklist de entrega do Tech Challenge | 🔲 | |

---

## Resumo de Progresso por Fase

| Fase | Concluídos | Pendentes | Aguardando decisão |
|---|---|---|---|
| 1 — Kickoff | 11 | 0 | 4 |
| 2 — Dados | 0 | 8 | 0 |
| 3 — Modelagem | 0 | 7 | 0 |
| 4 — MLOps | 0 | 7 | 0 |
| 5 — Entrega | 0 | 4 | 0 |
| **Total** | **11** | **26** | **4** |
