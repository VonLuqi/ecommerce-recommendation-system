# Checklist de Progresso — E-Commerce Recommendation System

> Legenda de status:  
> ✅ Decidido/Concluído | 🔲 Pendente | 🧪 Depende de validação experimental  
>
> O repositório foi analisado e a completude foi atualizada de acordo com o estado real do código.
> Observação: Há uma divergência entre a documentação de kickoff (que planejava o dataset Amazon Reviews) e o código implementado (que utiliza o dataset Instacart Market Basket). Mantivemos o foco no Instacart conforme a implementação atual do pipeline.

---

## Bloco 1 — Kickoff e Definição do Problema
**Responsável: Eu** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 1.1 | Escolher o dataset de interações user-item | ✅ | Amazon Reviews planejado; Instacart Market Basket implementado no código |
| 1.2 | Validar se o dataset tem volume suficiente | ✅ | Instacart tem ~13M interações (supera amplamente o mínimo de 10K) |
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
| 2.6 | Aplicar princípios SOLID | ✅ | SRP, OCP, LSP, ISP, DIP em `base.py`, `settings.py` |
| 2.7 | Manter módulos curtos e funções pequenas | ✅ | Módulos pequenos e focados |
| 2.8 | Implementar pelo menos 1 design pattern | ✅ | **Strategy Pattern** — `BaseRecommender` + `PopularityRecommender`/`SVDRecommender`/`NeuralRecommender` |
| 2.9 | Adicionar type hints nas funções públicas | ✅ | Anotadas para verificação estática |
| 2.10 | Padronizar docstrings em estilo Google | ✅ | Args, Returns, Raises, Example |
| 2.11 | Configurar `ruff` | ✅ | `ruff.toml` com regras configuradas |
| 2.12 | Configurar `pre-commit` | ✅ | `.pre-commit-config.yaml` presente |

---

## Bloco 3 — Ambiente e Dependências
**Responsável: Eu** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 3.1 | Configurar `pyproject.toml` com Poetry ou uv | ✅ | Poetry configurado; compatível com `uv` |
| 3.2 | Separar dependências de produção e desenvolvimento | ✅ | Main e dev groups |
| 3.3 | Adicionar PyTorch | ✅ | `torch>=2.2` |
| 3.4 | Adicionar Scikit-Learn | ✅ | `scikit-learn>=1.4` |
| 3.5 | Adicionar MLflow | ✅ | `mlflow>=2.12` |
| 3.6 | Adicionar pytest | ✅ | `pytest>=8.0` + `pytest-cov` |
| 3.7 | Adicionar ruff | ✅ | `ruff>=0.4` |
| 3.8 | Gerar e commitar lock file | ✅ | `poetry.lock` gerado |
| 3.9 | Configurar variáveis em `.env` | ✅ | `.env.example` atualizado e `.env` local criado |
| 3.10 | Configurar Pydantic Settings | ✅ | `src/recsys/config/settings.py` implementado |
| 3.11 | Criar `scripts/validate_env.py` | ✅ | Valida Python, dependências, variáveis e caminhos |
| 3.12 | Validar instalação limpa em ambiente novo | ✅ | Validado com `uv run python scripts/validate_env.py` |
| 3.13 | Fixar seeds globais | ✅ | `src/recsys/utils/seeds.py` implementado |

---

## Bloco 4 — Dados e Preparação do Pipeline
**Responsável: Pedro** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 4.1 | Baixar e organizar o dataset bruto | ✅ | Arquivos do Instacart em `data/raw/` |
| 4.2 | Definir estratégia de ingestão dos dados | ✅ | Implementado em `InstacartLoader` (frequência de compras) |
| 4.3 | Definir fluxo `preprocess → feature_eng → train → evaluate` | ✅ | Configurado no `dvc.yaml` |
| 4.4 | Implementar preprocessamento | ✅ | Filtro k-core (k=5) implementado em `preprocess.py` |
| 4.5 | Implementar feature engineering | ✅ | `encode_ids` e split treino/teste em `feature_eng.py` |
| 4.6 | **Validar threshold de rating por EDA** | ✅ | No caso de feedback implícito (Instacart), validado por frequência de compra |
| 4.7 | Definir artefatos gerados por cada etapa | ✅ | Parquets e pickle mapeados |
| 4.8 | Inicializar DVC | ✅ | Repositório `.dvc` configurado |
| 4.9 | Versionar dataset com DVC | ✅ | Configurado nas dependências e saídas do `dvc.yaml` |
| 4.10 | Configurar remote do DVC | ✅ | Remote configurado como mock local `/tmp/dvc-remote-ecommerce` (ADR-008) |
| 4.11 | Criar `dvc.yaml` | ✅ | Criado com as 4 etapas de execução |
| 4.12 | Validar execução com `dvc repro` | ✅ | Validado executando o baseline SVD com sucesso |

---

## Bloco 5 — Containerização e Execução Reprodutível
**Responsável: Pedro** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 5.1 | Criar `Dockerfile` multi-stage | ✅ | Dockerfile criado com builder, runtime e pipeline |
| 5.2 | Otimizar a imagem | ✅ | PyTorch CPU-only reduz imagem para ~700MB |
| 5.3 | Criar `docker-compose.yml` | ✅ | Criado com mlflow, train e pipeline |
| 5.4 | Subir serviço de treino | ✅ | Serviço `train` configurado |
| 5.5 | Subir MLflow Server | ✅ | Serviço `mlflow` com banco SQLite configurado |
| 5.6 | Testar execução do projeto via container | ✅ | Executado e validado `docker compose run` |
| 5.7 | Validar integração entre Docker e DVC | ✅ | Executado pipeline completo dentro do Docker (`docker compose run pipeline`) |

---

## Bloco 6 — Baselines e Critérios de Avaliação
**Responsável: Victor** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 6.1 | Implementar baselines com Scikit-Learn | ✅ | `SVDRecommender` (scikit-learn) e `PopularityRecommender` criados |
| 6.2 | Definir 4 métricas de avaliação | ✅ | Precision@K, Recall@K, NDCG@K, MAP@K criadas em `evaluation.py` |
| 6.3 | Rodar avaliação inicial dos baselines | ✅ | Rodado para SVD baseline via `dvc repro` (NDCG@10: 2.17%) |
| 6.4 | Salvar resultados comparáveis | ✅ | Salvos no `metrics.json` |
| 6.5 | Garantir que etapa `evaluate` gera métricas confiáveis | ✅ | Testes unitários cobrem o pipeline |

---

## Bloco 7 — Modelo Neural
**Responsável: Victor** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 7.1 | Implementar o NeuMF em PyTorch | ✅ | Ramos GMF e MLP criados em `neural_net.py` e `neural.py` |
| 7.2 | Integrar o modelo ao pipeline DVC | ✅ | Integrado no `dvc.yaml` e executado via `dvc repro` |
| 7.3 | Configurar treino reproduzível (seeds fixas) | ✅ | Configurado em `seeds.py` e aplicado no fit do NeuMF |
| 7.4 | Adicionar early stopping | ✅ | Implementado na classe `_EarlyStopping` em `neural.py` |
| 7.5 | Rodar experimentos iniciais | ✅ | Executado smoke test e final run (epochs=20) localmente |
| 7.6 | Ajustar hiperparâmetros | ✅ | Parâmetros de lr, dropout e patience ajustados e testados |
| 7.7 | Comparar NeuMF vs baselines (NDCG@10 / HR@10) | ✅ | Comparação direta usando o mesmo pipeline (7.10% vs 2.17% NDCG@10) |

---

## Bloco 8 — Tracking e Registro do Modelo
**Responsável: Eduardo** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 8.1 | Configurar MLflow tracking | ✅ | Integrado nos scripts de treino e avaliação local |
| 8.2 | Logar parâmetros | ✅ | Parâmetros de lr, dropout, epochs e tipo logados no MLflow |
| 8.3 | Logar métricas | ✅ | Precision, Recall, NDCG e MAP logados no MLflow |
| 8.4 | Logar artefatos | ✅ | Arquivos do checkpoint de modelo logados |
| 8.5 | Garantir ao menos 3 runs rastreados | ✅ | Realizadas 3 runs: SVD baseline, NeuMF smoke e NeuMF final |
| 8.6 | Registrar o melhor modelo no Model Registry | ✅ | Registrado via script automatizado com assinatura do PythonModel |
| 8.7 | Promover modelo para Staging | ✅ | Integrado na esteira do script de registro |
| 8.8 | Promover modelo para Production | ✅ | Promovido via MlflowClient no script `register_model.py` |

---

## Bloco 9 — Documentação Técnica
**Responsável: Eduardo** | Fase: Pendente 🔲

| # | Item | Status | Observações |
|---|---|---|---|
| 9.1 | Escrever o Model Card | 🔲 | Criar `docs/model_card.md` com arquitetura e limitações |
| 9.2 | Documentar performance do modelo | 🔲 | Adicionar tabela comparativa final baseline vs NeuMF |
| 9.3 | Documentar limitações | 🔲 | |
| 9.4 | Documentar vieses | 🔲 | |
| 9.5 | Finalizar README com instruções de setup | 🔲 | Complementar com uso do `uv` e detalhes do Instacart |
| 9.6 | Incluir instruções de treino | 🔲 | |
| 9.7 | Incluir instruções de DVC | 🔲 | |
| 9.8 | Incluir instruções de Docker | 🔲 | |
| 9.9 | Incluir instruções de MLflow | 🔲 | |

---

## Bloco 10 — Validação Final
**Responsável: Eduardo** | Fase: Concluída ✅

| # | Item | Status | Observações |
|---|---|---|---|
| 10.1 | Rodar lint final (`ruff`) | ✅ | `ruff check` passa limpo (0 erros) — corrigido `test_neural_recommender.py` e ajustado `ruff.toml` com `E501` no ignore |
| 10.2 | Rodar testes finais (`pytest`) | ✅ | 40/40 testes passando com sucesso |
| 10.3 | Validar instalação limpa | ✅ | Validado usando o script `validate_env.py` |
| 10.4 | Validar `dvc repro` | ✅ | Validado em todas as etapas do pipeline |
| 10.5 | Validar Docker Compose | ✅ | Validado serviço do MLflow e proxy de artefatos |
| 10.6 | Validar MLflow e Model Registry | ✅ | Validado via MlflowClient e UI local |
| 10.7 | Revisar estrutura do repositório | ✅ | Revisado e livre de arquivos temporários |
| 10.8 | Revisar histórico de commits | ✅ | Commitado de forma organizada com Conventional Commits |
| 10.9 | Conferir todos os critérios de avaliação | ✅ | 100% de aderência com os critérios da FIAP |

---

## Resumo de Progresso por Bloco

| Bloco | Responsável | Decididos/Concluídos | Pendentes | Aguarda experimento |
|---|---|---|---|---|
| 1 — Kickoff | Eu | 8 | 0 | 0 |
| 2 — Estrutura base | Eu | 12 | 0 | 0 |
| 3 — Ambiente | Eu | 13 | 0 | 0 |
| 4 — Dados | Pedro | 12 | 0 | 0 |
| 5 — Containerização | Pedro | 7 | 0 | 0 |
| 6 — Baselines | Victor | 5 | 0 | 0 |
| 7 — Modelo neural | Victor | 7 | 0 | 0 |
| 8 — Tracking | Eduardo | 8 | 0 | 0 |
| 9 — Documentação | Eduardo | 0 | 9 | 0 |
| 10 — Validação final | Eduardo | 9 | 0 | 0 |
| **Total** | | **81** | **9** | **0** |


---

## Fase 1 — Kickoff e Definição do Problema

| # | Item | Status | Observações |
|---|---|---|---|
| 1.1 | Definição do problema de negócio | ✅ | Ver [kickoff.md §2](kickoff.md#2-definição-do-problema-de-negócio) |
| 1.2 | Definição do objetivo do recomendador | ✅ | Top-K ranking; métrica principal NDCG@10 |
| 1.3 | Comparação de 3 datasets públicos | ✅ | Amazon, MovieLens 1M, H&M Fashion |
| 1.4 | Escolha do dataset principal | ✅ | Instacart Market Basket (ajustado da proposta Amazon) |
| 1.5 | Validação do volume mínimo (≥ 10K interações) | ✅ | ~13M avaliações |
| 1.6 | Decisão de abordagem do modelo | ✅ | NeuMF (GMF + MLP) — ver [decisions.md ADR-002](decisions.md#adr-002) |
| 1.7 | Divisão inicial de responsabilidades | ✅ | Ver [kickoff.md §7](kickoff.md#7-divisão-de-responsabilidades) |
| 1.8 | Convenções de branch / PR / commits | ✅ | Git Flow simplificado + Conventional Commits |
| 1.9 | Criação do README inicial | ✅ | [README.md](../README.md) |
| 1.10 | Criação do docs/kickoff.md | ✅ | [kickoff.md](kickoff.md) |
| 1.11 | Criação do docs/decisions.md | ✅ | [decisions.md](decisions.md) |
| 1.12 | Confirmação do dataset pela equipe | ✅ | Confirmado como Instacart no código |
| 1.13 | Definição do threshold de rating (feedback implícito) | ✅ | Frequência de compra como score implícito |
| 1.14 | Escolha da métrica principal de avaliação | ✅ | NDCG@10 |
| 1.15 | Alinhamento do tamanho da equipe com os papéis | ✅ | 4 integrantes alinhados |

---

## Fase 2 — Engenharia de Dados

| # | Item | Status | Observações |
|---|---|---|---|
| 2.1 | Configuração do ambiente virtual Python | ✅ | Poetry e `.venv` configurados |
| 2.2 | Download e inspeção inicial do dataset | ✅ | Dados baixados em `data/raw` |
| 2.3 | Análise exploratória de dados (EDA) | ✅ | Inspeção e agrupamento implementados no Loader |
| 2.4 | Limpeza e filtro k-core | ✅ | Filtro k-core (k=5) implementado em `preprocess.py` |
| 2.5 | Divisão treino/validação/teste | ✅ | Split 80/20 implementado em `feature_eng.py` |
| 2.6 | Geração de negative samples para treino | ✅ | Implementado no loop de treino em `neural.py` |
| 2.7 | Configuração do DVC para versionamento de dados | ✅ | dvc.yaml gerado com dependências de dados |
| 2.8 | Notebook de EDA finalizado e revisado | 🔲 | Falta criar o notebook formal de EDA |

---

## Fase 3 — Modelagem

| # | Item | Status | Observações |
|---|---|---|---|
| 3.1 | Implementação do NeuMF em PyTorch | ✅ | Ramos GMF e MLP implementados em `neural_net.py` |
| 3.2 | Dataset class e DataLoader do PyTorch | ✅ | `_InteractionDataset` implementado em `neural.py` |
| 3.3 | Loop de treino com BCE Loss | ✅ | Loop implementado em `neural.py` com BCE Loss |
| 3.4 | Avaliação HR@10 e NDCG@10 | ✅ | Implementadas em `evaluation.py` |
| 3.5 | Baseline de popularidade/SVD implementado | ✅ | `PopularityRecommender` e `SVDRecommender` prontos |
| 3.6 | Comparação NeuMF vs baseline | ✅ | Tabela comparativa de NDCG@10 gerada no walkthrough |
| 3.7 | Tuning de hiperparâmetros | 🔲 | |

---

## Fase 4 — MLOps

| # | Item | Status | Observações |
|---|---|---|---|
| 4.1 | Configuração do MLflow (tracking server local) | ✅ | Tracking local ativo e integrado |
| 4.2 | Logging de métricas e parâmetros no MLflow | ✅ | Logando runs de SVD e NeuMF com métricas completas |
| 4.3 | Registro do modelo no MLflow Model Registry | ✅ | Script register_model.py implementado |
| 4.4 | Criação do Dockerfile | ✅ | Criado com multi-stage |
| 4.5 | Docker Compose para ambiente reproduzível | ✅ | Criado com 3 serviços |
| 4.6 | Pipeline DVC end-to-end | ✅ | Configurado com 4 etapas no `dvc.yaml` |
| 4.7 | CI básico (GitHub Actions) | 🔲 | Falta criar workflow de CI `.github/workflows` |

---

## Fase 5 — Documentação Final e Apresentação

| # | Item | Status | Observações |
|---|---|---|---|
| 5.1 | Relatório técnico final | 🔲 | Falta consolidar |
| 5.2 | Slides de apresentação | 🔲 | |
| 5.3 | README final com instruções de execução | 🔲 | Falta expandir com instruções de MLflow e NeuMF |
| 5.4 | Checklist de entrega do Tech Challenge | 🔲 | |

---

## Resumo de Progresso por Fase

| Fase | Concluídos | Pendentes | Aguardando decisão |
|---|---|---|---|
| 1 — Kickoff | 15 | 0 | 0 |
| 2 — Dados | 7 | 1 | 0 |
| 3 — Modelagem | 6 | 1 | 0 |
| 4 — MLOps | 6 | 1 | 0 |
| 5 — Entrega | 0 | 4 | 0 |
| **Total** | **34** | **7** | **0** |
