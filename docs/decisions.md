# Decisões Arquiteturais — E-Commerce Recommendation System

> Registro de decisões técnicas relevantes no formato ADR (Architecture Decision Record) simplificado.  
> Cada entrada documenta o contexto, a decisão tomada, as alternativas consideradas e as consequências esperadas.

---

## ADR-001 — Dataset: Amazon Reviews 2023 "Toys and Games"

| Campo | Detalhe |
|---|---|
| **Status** | Aceito e confirmado pela equipe |
| **Data** | Junho de 2026 |
| **Responsável** | Equipe — fase de kickoff |

### Contexto

O projeto requer um dataset com no mínimo 10.000 interações user-item, de domínio preferencialmente relacionado a e-commerce, disponível publicamente para uso acadêmico.

### Decisão

Utilizar o subconjunto **"Toys_and_Games"** do dataset **Amazon Reviews 2023** (McAuley Lab, UCSD).

Fonte: `McAuley-Lab/Amazon-Reviews-2023` via HuggingFace Datasets.

### Alternativas Consideradas

| Alternativa | Motivo de descarte |
|---|---|
| MovieLens 1M | Domínio de filmes — distância semântica do e-commerce prejudica a narrativa do projeto |
| H&M Personalized Fashion | Volume excessivo (31 M transações) e licença Kaggle restritiva |

### Consequências

- **Positivas:** Domínio e-commerce real; metadados ricos; reproduzível sem autenticação; referência acadêmica.
- **Negativas:** Esparsidade alta (~99%) — requer filtro k-core antes do treino.
- **Risco:** Subconjunto pode ter distribuição de ratings enviesada para notas altas (problema comum em reviews voluntárias). Mitigação: EDA detalhada na fase de dados.

---

## ADR-002 — Arquitetura do Modelo: NeuMF (Neural Collaborative Filtering)

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Equipe — fase de kickoff |

### Contexto

O desafio exige uma rede neural do tipo MLP ou embedding-based. É necessário escolher uma arquitetura que seja implementável em PyTorch, avaliável com métricas padrão (HR@K, NDCG@K) e que produza resultados publicáveis para comparação.

### Decisão

Implementar **NeuMF (Neural Matrix Factorization)**, conforme proposto em:

> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). *Neural Collaborative Filtering*. Proceedings of WWW '17.

A arquitetura combina dois ramos paralelos:
- **GMF (Generalized Matrix Factorization):** produto elementar de embeddings de usuário e item.
- **MLP:** concatenação de embeddings de usuário e item passada por camadas densas com ativação ReLU.

Os dois ramos são fundidos em uma camada final com ativação sigmoid para classificação binária (interação positiva / negativa).

### Alternativas Consideradas

| Alternativa | Motivo de descarte |
|---|---|
| MLP puro | Menor expressividade — não captura fatores latentes tão bem quanto abordagens baseadas em embeddings |
| Matrix Factorization clássico (SVD) | Não utiliza PyTorch de forma nativa; menor flexibilidade para extensões futuras |
| BERT4Rec / modelos sequenciais | Complexidade desnecessária para o escopo do projeto; requer datasets com dados sequenciais explícitos |

### Consequências

- **Positivas:** Satisfaz ambos os requisitos (MLP + embeddings); benchmark publicado; extensível para side information.
- **Negativas:** Duas tabelas de embedding separadas por ramo aumentam ligeiramente o número de parâmetros.
- **Risco:** Overfitting em datasets esparsos. Mitigação: dropout nos ramos MLP e regularização L2 nos embeddings.

---

## ADR-003 — Estratégia de Feedback: Implícito com Threshold em Ratings

| Campo | Detalhe |
|---|---|
| **Status** | Proposto — aguarda validação por EDA *(responsável: Pedro, bloco 4)* |
| **Data** | Junho de 2026 |
| **Responsável** | Pedro (fase de dados) |

### Contexto

O dataset Amazon Reviews contém ratings explícitos de 1 a 5 estrelas. O NeuMF, na formulação original de He et al. (2017), é treinado com feedback implícito binário (interagiu / não interagiu). É necessário definir como converter os ratings em sinal de treino.

### Decisão Proposta

Converter ratings em feedback implícito binário com threshold:
- **Positivo (label = 1):** rating ≥ 4
- **Negativo (label = 0):** gerado por negative sampling uniforme (itens não vistos pelo usuário)

Razão do threshold em 4: ratings 1–3 indicam insatisfação ou indiferença — incluí-los como positivos introduziria ruído no sinal de treinamento.

### Alternativas a Avaliar

| Alternativa | Observação |
|---|---|
| Qualquer rating como positivo (threshold = 1) | Máximo de dados positivos; ruído elevado por incluir avaliações negativas |
| Threshold em 3 | Intermediário; pode ser considerado se a distribuição de ratings for muito enviesada para 4–5 |
| Apenas ratings = 5 | Sinal mais puro; volume menor de positivos; risco de underfitting |

### Ação Requerida

- [ ] Pedro deve realizar EDA da distribuição de ratings (bloco 4 do plano) antes de confirmar o threshold.
- [ ] Caso a distribuição seja muito enviesada para 4–5, reconsiderar threshold em 3.
- [ ] Registrar aqui a decisão final com o histograma de distribuição como evidência.

---

## ADR-004 — Convenção de Versionamento: Git Flow Simplificado

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Equipe — fase de kickoff |

### Contexto

É necessário definir uma estratégia de branches e PR que equilibre organização e agilidade para uma equipe acadêmica de tamanho reduzido.

### Decisão

Adotar Git Flow simplificado com duas branches permanentes (`main` e `develop`) e branches de vida curta por escopo (`feature/`, `model/`, `fix/`, `docs/`, `chore/`).

Commits seguem o padrão **Conventional Commits** para geração futura de changelogs automatizados.

### Consequências

- **Positivas:** Organização clara do histórico; facilita code review; compatível com CI/CD futuro.
- **Negativas:** Overhead leve de processo para equipes pequenas.
- **Adaptação para equipe de 4 pessoas:** O PR de `develop` → `main` requer aprovação de pelo menos 1 outro integrante além do autor.

---

## ADR-005 — Métrica Principal de Avaliação: NDCG@10

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Equipe — fase de kickoff |

### Contexto

Duas métricas são padrão para avaliação de sistemas de recomendação: HR@K (Hit Rate) e NDCG@K (Normalized Discounted Cumulative Gain). É necessário definir qual será o número titular da apresentação final.

### Decisão

Adotar **NDCG@10** como métrica principal (headline number). **HR@10** será reportado como métrica secundária e complemento narrativo para audiência não-técnica.

### Justificativa Técnica

| Fator | HR@10 | NDCG@10 | Decisão |
|---|---|---|---|
| Sensibilidade à posição do item | Não | Sim | NDCG@10 ✔ |
| Paper de referência (He et al., 2017) | Secundário | Primário | NDCG@10 ✔ |
| Poder discriminativo entre modelos | Menor | Maior | NDCG@10 ✔ |
| Resistência a inflação por popularidade | Menor | Maior | NDCG@10 ✔ |
| Facilidade de explicação | Alta | Média | HR@10 ✔ |

**Conclusão:** NDCG@10 é mais rigoroso, mais discriminativo e diretamente comparável com os resultados publicados no paper do NeuMF. HR@10 é mais fácil de comunicar, por isso será reportado como suporte narrativo nas apresentações.

### Alternativas Consideradas

| Alternativa | Motivo de descarte como primário |
|---|---|
| HR@10 como única métrica | Insensível à posição; baseline de popularidade pode inflar artificialmente o resultado |
| MAP@10 | Menos comum na literatura de NCF; mais complexo sem ganho adicional |
| Precision@10 / Recall@10 | Adequados para conjuntos com múltiplos itens relevantes; menos padrão no protocolo leave-one-out |
