# Decisões Arquiteturais — E-Commerce Recommendation System

> Registro de decisões técnicas relevantes no formato ADR (Architecture Decision Record) simplificado.  
> Cada entrada documenta o contexto, a decisão tomada, as alternativas consideradas e as consequências esperadas.

---

## ADR-001 — Dataset: Instacart Market Basket Analysis

| Campo | Detalhe |
|---|---|
| **Status** | Aceito e confirmado pela equipe (Atualizado) |
| **Data** | Junho de 2026 |
| **Responsável** | Equipe — fase de kickoff |

### Contexto

O projeto requer um dataset com no mínimo 10.000 interações user-item, de domínio preferencialmente relacionado a e-commerce, disponível publicamente para uso acadêmico. Inicialmente planejou-se usar o Amazon Reviews 2023 "Toys and Games", mas a base de código e os testes foram construídos em cima do Instacart Market Basket.

### Decisão

Utilizar o dataset **Instacart Market Basket Analysis** (~3M pedidos, ~200K usuários, ~50K produtos). 
A frequência de compra de cada par (usuário, item) no histórico `order_products__prior.csv` será tratada como rating implícito.

### Alternativas Consideradas

| Alternativa | Motivo de descarte |
|---|---|
| Amazon Reviews 2023 "Toys and Games" | Descartado para evitar retrabalho, pois a infraestrutura do Instacart já estava 100% implementada e testada |
| MovieLens 1M | Domínio de filmes — distância semântica do e-commerce |

### Consequências

- **Positivas:** Infraestrutura, loaders e pipeline de dados já operacionais e testados; base de dados muito maior (~13M interações úteis).
- **Negativas:** Requer download manual via Kaggle e cópia para o diretório `data/raw/` devido a restrições de autenticação da API.

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

### Consequências

- **Positivas:** Satisfaz ambos os requisitos (MLP + embeddings); benchmark publicado; extensível para side information.
- **Negativas:** Duas tabelas de embedding separadas por ramo aumentam ligeiramente o número de parâmetros.

---

## ADR-003 — Estratégia de Feedback Implícito e Amostragem de Negativos

| Campo | Detalhe |
|---|---|
| **Status** | Aceito (Atualizado) |
| **Data** | Junho de 2026 |
| **Responsável** | Pedro e Victor |

### Contexto

O Instacart Market Basket fornece apenas interações positivas (compras ocorridas). Para o treinamento do NeuMF com `Binary Cross-Entropy (BCE) Loss`, necessitamos de dados de feedback implícito binários (interação positiva = 1, ausência de interação = 0).

### Decisão

1. **Interações Positivas (label = 1):** Definir as compras ocorridas (frequência acumulada) de cada par usuário-item como sinal positivo.
2. **Amostragem de Negativos (label = 0):** Para cada interação positiva do usuário, amostrar uniformemente $M = 4$ itens do catálogo completo de produtos com os quais o usuário **nunca** interagiu.
3. **Geração Dinâmica:** A amostragem de negativos deve ser gerada dinamicamente a cada época no início do treino para evitar overfitting em amostras estáticas (conforme preconizado por He et al., 2017).

### Alternativas Consideradas

| Alternativa | Motivo de descarte |
|---|---|
| Amostragem Estática de Negativos | Menor variabilidade do sinal negativo, levando a overfitting rápido |
| Sem Negativos (Treino Apenas com Positivos) | Impossibilita o cálculo da BCE Loss binária |

### Consequências

- **Positivas:** Sinal de treino robusto, menor overfitting e alta conformidade com o artigo do NeuMF.
- **Negativas:** Leve overhead computacional no início de cada época para realizar a busca/amostragem de itens não comprados.

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

Adotar **NDCG@10** como métrica principal (headline number). **HR@10** será reportado como métrica secundária.

---

## ADR-006 — Otimização do Cálculo do NDCG@K

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Victor / Eduardo |

### Contexto

A etapa de avaliação (`evaluate.py`) roda em CPU e avalia as recomendações sobre 200.000 usuários de teste. O uso de `sklearn.metrics.ndcg_score` dentro de um loop iterativo em Python introduz um overhead massivo de validação, fazendo com que a avaliação demore de 8 a 10 minutos.

### Decisão

Implementar uma função matemática vetorizada direta para calcular o NDCG@K binário em Python/NumPy puro. A relevância ideal (IDCG) para relevância binária pode ser calculada usando a soma acumulada de coeficientes logarítmicos e comparando o número total de itens relevantes.

### Consequências

- **Positivas:** Redução do tempo de avaliação de ~10 minutos para menos de 10 segundos (ganho de performance de ~100x).
- **Negativas:** Lógica matemática customizada que precisa ser testada unitariamente para garantir equivalência matemática com a biblioteca oficial.

---

## ADR-007 — Carregamento Dinâmico e Unificado de Modelos no Pipeline

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Eduardo |

### Contexto

A etapa de treino (`train.py`) pode gerar dois tipos de arquivos de pesos muito distintos:
- Baseline SVD: Objeto Python serializado via `pickle` (extensão `.pkl`).
- Modelo PyTorch NeuMF: Checkpoint contendo state_dict e mappings serializado via `torch.save` (extensão `.pth`).
O script `evaluate.py` precisa processar ambos de forma automática e transparente.

### Decisão

Implementar no `evaluate.py` uma função `load_model(model_path: Path) -> BaseRecommender` que verifica a extensão do arquivo do modelo:
- Se `.pth`, importa `NeuralRecommender`, instancia a classe e chama seu método `.load(model_path)`.
- Se `.pkl`, carrega via biblioteca nativa `pickle`.

### Consequências

- **Positivas:** Pipeline unificado e transparente; mantém o design pattern Strategy intacto.
- **Negativas:** Importação tardia do PyTorch no evaluate para evitar carregamento pesado desnecessário quando rodando apenas baselines em CPU simples.

---

## ADR-008 — Mock do Remote do DVC

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Pedro |

### Contexto

O repositório possui uma configuração de DVC remote que aponta para um caminho local absoluto do Windows de um membro específico da equipe. Para possibilitar o uso de comandos como `dvc push` e `dvc pull` em ambiente local multiplataforma (Mac/Linux/Windows) sem requerer armazenamento pago em nuvem, precisamos de um mock local.

### Decisão

Configurar um remote DVC local que aponte para uma pasta temporária padrão fora do repositório (ex: `/tmp/dvc-remote-ecommerce`). Isso garante compatibilidade multiplataforma em ambientes Unix e Windows (através de mapeamento compatível).

### Consequências

- **Positivas:** Comandos de DVC como `dvc push`/`pull` funcionam localmente sem falhar por falhas de caminho absoluto do Windows de terceiros.
- **Negativas:** Os dados versionados não são persistidos de fato em um servidor em nuvem externo compartilhado, apenas simulados localmente.

---

## ADR-009 — Recomendações em Lote (Batch) no Pipeline de Avaliação

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Eduardo |

### Contexto

O pipeline de avaliação (`evaluate.py`) roda sobre o dataset Instacart de teste contendo ~200 mil usuários únicos. A geração de recomendações de forma individual (iteração sequencial via loop Python chamando `model.recommend`) causava um overhead massivo de processamento, pois:
1. No SVD, realizava a ordenação completa de todos os ~50 mil itens do catálogo via `argsort` (complexidade $O(N \log N)$) 200 mil vezes consecutivas.
2. No NeuMF, invocava a inferência do modelo do PyTorch 200 mil vezes, gerando altíssimo overhead de execução no loop Python e sob uso sustentado de 100% de CPU.
3. Não aproveitava de forma eficiente as instruções vetorizadas (SIMD) de baixo nível ou aceleração por hardware (MPS/Metal no Mac do usuário).

### Decisão

1. Estender a interface `BaseRecommender` com o método `recommend_batch(self, user_ids: list[Any], top_k: int = 10) -> dict[Any, list[str]]`.
2. Implementar no `SVDRecommender` o processamento em blocos (batches de 5.000 usuários), vetorizando o produto interno com toda a matriz de itens componentes em uma única chamada do NumPy (`np.dot`).
3. Substituir `np.argsort` por `np.argpartition` para selecionar de forma linear $O(N)$ apenas os top-K itens de maior interesse de cada usuário, ordenando somente essa pequena porção final.
4. Implementar no `NeuralRecommender` o processamento em blocos (batches de 256 usuários), repetindo os tensores de entrada para todos os itens do catálogo e executando a inferência no PyTorch em lote.
5. Adicionar detecção automática do backend MPS (`mps` - Metal Performance Shaders) do PyTorch para permitir aceleração nativa por GPU no processador Apple Silicon (M4 Pro) do desenvolvedor no host.
6. Alterar a lógica do `evaluate.py` para invocar o método `recommend_batch` com todos os usuários do conjunto de teste em uma única chamada.

### Consequências

- **Positivas:** 
  - Redução drástica da latência de avaliação (de minutos para poucos segundos).
  - Queda significativa de consumo de CPU e uso eficiente de recursos.
  - Habilidade de utilizar a GPU nativa do Apple Silicon via MPS se executado fora do container Docker.
- **Negativas:**
  - Aumento temporário do consumo de memória RAM durante o produto matricial de lote (controlado definindo tamanhos máximos de bloco de 5000 para NumPy e 256 para PyTorch).

---

## ADR-010 — Aceleração de Hardware por GPU (MPS / CUDA)

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Eduardo |

### Contexto

O modelo de recomendação neural (NeuMF) envolve o treinamento e a inferência sobre mais de 10 milhões de interações de dados implícitos. O processamento dessas operações em CPU é demorado. Como a máquina de desenvolvimento local é um Apple Silicon (Mac M4 Pro) e o ambiente de produção/nuvem pode conter GPUs NVIDIA, o sistema deve ser capaz de utilizar aceleração por hardware (GPU) de forma dinâmica e transparente se disponível.

### Decisão

1. Implementar detecção inteligente do backend de execução em `NeuralRecommender`:
   ```python
   if torch.cuda.is_available():
       self.device = torch.device("cuda")
   elif torch.backends.mps.is_available():
       self.device = torch.device("mps")
   else:
       self.device = torch.device("cpu")
   ```
2. Garantir que todas as alocações de tensores, inicialização do modelo (`NeuMF`) e passagem de lotes no DataLoader de validação/treino respeitem e enviem os dados para o device selecionado dinamicamente.
3. No container Docker, devido a restrições de virtualização do macOS que não repassam o driver Metal (MPS) diretamente ao container Linux, o PyTorch utiliza fallback automático para CPU de forma resiliente e sem quebras de execução.

### Consequências

- **Positivas:** Execução extremamente rápida se rodado nativamente no host com GPU (MPS) ou em ambientes em nuvem com GPUs NVIDIA (CUDA).
- **Negativas:** Nenhuma perceptível, já que o fallback silencioso para `cpu` garante o funcionamento normal dentro do Docker.

---

## ADR-011 — Prevenção de Out-Of-Memory (OOM) via Redução do Lote de Avaliação do NeuMF

| Campo | Detalhe |
|---|---|
| **Status** | Aceito |
| **Data** | Junho de 2026 |
| **Responsável** | Eduardo |

### Contexto

Durante a inferência em lote (`recommend_batch`) no `NeuralRecommender` para gerar as recomendações de avaliação (200 mil usuários sobre o catálogo total de 46.368 itens), o uso de um lote de 256 usuários exigia a expansão de tensores cartesianos de entrada.
Isso resultava em tensores de $256 \times 46.368 = 11.870.208$ linhas de entrada. A primeira camada linear do bloco MLP (`Linear(16, 256)`) gerava uma matriz de ativação gigante de $11,8 \times 10^6 \times 256$ floats, necessitando de **12.15 GB** de memória contígua na RAM. Isso causava o encerramento imediato do container Docker pelo kernel do Linux (`OOM-killed` com código 137).

### Decisão

Reduzir o tamanho do lote de processamento de usuários (`batch_size`) na geração de recomendações no `NeuralRecommender` de **256** para **16**.

### Consequências

- **Positivas:** 
  - Redução drástica da memória consumida pelas ativações da camada linear para apenas ~380 MB por lote (completamente segura para rodar em containers limitados).
  - Estabilidade e sucesso total na conclusão do pipeline DVC de ponta a ponta.
  - O overhead de CPU do loop do Python para blocos de tamanho 16 continua insignificante, permitindo que a etapa inteira de avaliação leve menos de 10 segundos.
- **Negativas:**
  - Menor grau de paralelização em placas de vídeo de alta capacidade (GPUs dedicadas grandes), mas sem impacto significativo no tempo total do pipeline atual.

