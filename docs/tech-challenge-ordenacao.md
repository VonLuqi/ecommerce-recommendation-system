# Tech Challenge — Ordem cronológica real

## 1. Kickoff e definição do problema
- [ ] Escolher o dataset de interações user-item
- [ ] Validar se o dataset tem volume suficiente para o desafio
- [ ] Definir o problema de negócio do e-commerce
- [ ] Definir o objetivo do sistema de recomendação
- [ ] Escolher a abordagem principal do modelo: MLP ou embedding-based
- [ ] Definir os integrantes e responsabilidades
- [ ] Criar o repositório GitHub
- [ ] Definir padrão de branch, PR e commits semânticos

## 2. Estrutura base do projeto
- [ ] Criar estrutura inicial com `src/`, `tests/`, `data/`, `models/` e `configs/`
- [ ] Configurar `.gitignore`
- [ ] Configurar `.dockerignore`
- [ ] Configurar `.env.example`
- [ ] Aplicar naming conventions
- [ ] Aplicar princípios SOLID
- [ ] Manter módulos curtos e funções pequenas
- [ ] Implementar pelo menos 1 design pattern
- [ ] Adicionar type hints nas funções públicas
- [ ] Padronizar docstrings em estilo Google
- [ ] Configurar `ruff`
- [ ] Configurar `pre-commit`

## 3. Ambiente e dependências
- [ ] Configurar `pyproject.toml` com Poetry ou uv
- [ ] Separar dependências de produção e desenvolvimento
- [ ] Adicionar PyTorch
- [ ] Adicionar Scikit-Learn
- [ ] Adicionar MLflow
- [ ] Adicionar pytest
- [ ] Adicionar ruff
- [ ] Gerar lock file
- [ ] Commitar lock file
- [ ] Configurar variáveis em `.env`
- [ ] Configurar Pydantic Settings
- [ ] Criar `scripts/validate_env.py`
- [ ] Validar instalação limpa em ambiente novo
- [ ] Fixar seeds globais

## 4. Dados e preparação do pipeline
- [ ] Baixar e organizar o dataset bruto
- [ ] Definir estratégia de ingestão dos dados
- [ ] Definir fluxo `preprocess -> feature_eng -> train -> evaluate`
- [ ] Implementar preprocessamento
- [ ] Implementar feature engineering
- [ ] Definir artefatos gerados por cada etapa
- [ ] Inicializar DVC
- [ ] Versionar dataset com DVC
- [ ] Configurar remote do DVC
- [ ] Criar `dvc.yaml`
- [ ] Validar execução com `dvc repro`

## 5. Containerização e execução reprodutível
- [ ] Criar `Dockerfile` multi-stage
- [ ] Otimizar a imagem
- [ ] Criar `docker-compose.yml`
- [ ] Subir serviço de treino
- [ ] Subir MLflow Server
- [ ] Testar execução do projeto via container
- [ ] Validar integração entre Docker e DVC

## 6. Baselines e critérios de avaliação
- [ ] Implementar baselines com Scikit-Learn
- [ ] Definir 4 métricas de avaliação
- [ ] Rodar avaliação inicial
- [ ] Salvar resultados comparáveis
- [ ] Garantir que a etapa `evaluate` do pipeline gera métricas confiáveis

## 7. Modelo neural
- [ ] Implementar o recomendador em PyTorch
- [ ] Integrar o modelo ao pipeline
- [ ] Configurar treino reproduzível
- [ ] Adicionar early stopping
- [ ] Rodar experimentos
- [ ] Ajustar hiperparâmetros
- [ ] Comparar o modelo neural com os baselines

## 8. Tracking e registro do modelo
- [ ] Configurar MLflow tracking
- [ ] Logar parâmetros
- [ ] Logar métricas
- [ ] Logar artefatos
- [ ] Garantir ao menos 3 runs rastreados
- [ ] Registrar o melhor modelo no Model Registry
- [ ] Promover modelo para Staging
- [ ] Promover modelo para Production

## 9. Documentação técnica
- [ ] Escrever o Model Card
- [ ] Documentar performance
- [ ] Documentar limitações
- [ ] Documentar vieses
- [ ] Finalizar README
- [ ] Incluir instruções de setup
- [ ] Incluir instruções de treino
- [ ] Incluir instruções de DVC
- [ ] Incluir instruções de Docker
- [ ] Incluir instruções de MLflow

## 10. Validação final
- [ ] Rodar lint final
- [ ] Rodar testes finais
- [ ] Validar instalação limpa
- [ ] Validar `dvc repro`
- [ ] Validar Docker Compose
- [ ] Validar MLflow e Model Registry
- [ ] Revisar estrutura do repositório
- [ ] Revisar histórico de commits
- [ ] Conferir todos os critérios de avaliação