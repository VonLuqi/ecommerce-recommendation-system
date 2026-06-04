# E-Commerce Recommendation System

![Fase](https://img.shields.io/badge/fase-kickoff-blue)
![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20MLflow%20%7C%20DVC%20%7C%20Docker-informational)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Sistema de recomendação de produtos para e-commerce baseado em comportamento de interação usuário-item, implementado com redes neurais do tipo embedding-based (NeuMF).

---

## Objetivo

Dado o histórico de interações de um usuário (cliques, compras, avaliações), ranquear produtos com maior probabilidade de engajamento, aumentando a taxa de conversão e reduzindo o abandono de sessão.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Modelagem | PyTorch, Scikit-Learn |
| Rastreamento de experimentos | MLflow |
| Versionamento de dados | DVC |
| Empacotamento | Docker |
| Linguagem | Python 3.11+ |

---

## Estrutura do Repositório

```
ecommerce-recommendation-system/
├── README.md
├── docs/
│   ├── kickoff.md          # Definição do problema, dataset, abordagem
│   ├── decisions.md        # Registro de decisões arquiteturais (ADRs)
│   └── checklist.md        # Checklist de progresso do projeto
├── data/                   # (próxima etapa — gerenciado via DVC)
├── notebooks/              # (próxima etapa — exploração e EDA)
├── src/                    # (próxima etapa — código-fonte)
└── experiments/            # (próxima etapa — MLflow runs)
```

---

## Documentação

- [Kickoff & Definição do Problema](docs/kickoff.md)
- [Decisões Arquiteturais](docs/decisions.md)
- [Checklist de Progresso](docs/checklist.md)

---

## Convenções de Desenvolvimento

### Branches

```
main                              → código estável/entregável
develop                           → integração contínua da equipe
feature/<escopo>/<descricao>      → ex: feature/data/amazon-loader
fix/<descricao>                   → ex: fix/embedding-shape
docs/<descricao>                  → ex: docs/kickoff-inicial
model/<descricao>                 → ex: model/neumf-architecture
```

### Commits Semânticos (Conventional Commits)

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

### Pull Requests

- PR obrigatório para `develop` e `main`
- Mínimo 1 revisor por PR
- Descrever: o que muda, como testar, checklist de qualidade

---

## Referências

- He, X. et al. (2017). *Neural Collaborative Filtering*. WWW '17.
- McAuley Lab, UCSD — *Amazon Reviews 2023 Dataset*. https://amazon-reviews-2023.github.io
