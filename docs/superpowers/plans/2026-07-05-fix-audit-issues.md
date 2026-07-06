# Correção dos Issues do Audit — Plano de Implementação

> **Para workers agênticos:** SUB-SKILL OBRIGATÓRIA: Use superpowers:subagent-driven-cdevelopment (recomendado) ou superpowers:executing-plans para implementar este plano tarefa por tarefa. Os passos usam sintaxe de checkbox (`- [ ]`) para rastreamento.

**Objetivo:** Corrigir os 5 issues mapeados no relatório de auditoria (`exploration_audit_plan.md`) do Tech Challenge Fase 02, sem alterar o comportamento funcional já validado.

**Arquitetura:** Cada issue é isolado e independente — não há dependência entre as tarefas. As correções tocam `neural.py` (parametrização + batching seguro), `train.py` (log de métricas), `settings.py` (validação de enum), e a integração de `popularity.py` no pipeline DVC/params.

**Tech Stack:** Python 3.11, PyTorch, MLflow, Pydantic Settings, DVC, pytest.

---

## Mapeamento de Issues → Tarefas

| # | Issue                                                  | Tarefa   |
| - | ------------------------------------------------------ | -------- |
| 1 | Hardcoded`num_negatives=4`                           | Tarefa 1 |
| 2 | OOM risk em`recommend_batch` (`repeat_interleave`) | Tarefa 2 |
| 3 | Sem`mlflow.log_metric` em `train.py`               | Tarefa 3 |
| 4 | `popularity.py` não integrado ao DVC/params         | Tarefa 4 |
| 5 | Sem enum constraint em`recommender_type`             | Tarefa 5 |

---

### Tarefa 1: Parametrizar `num_negatives` no NeuMF

**Arquivos:**

- Modificar: `src/recsys/recommenders/neural.py:74,303,376,450,464`
- Teste: `tests/test_neural_recommender.py`

- [ ] **Passo 1: Ler o estado atual das assinaturas afetadas**

Confirme os defaults hardcoded antes de editar:

```bash
grep -n "num_negatives" src/recsys/recommenders/neural.py
```

Esperado: ocorrências em `_NegativeSamplingCollate.__init__` (linha ~74), `_sample_negatives_dataset` (linha ~303), `_sample_negatives` (linha ~376), e duas chamadas dentro de `NeuralRecommender.fit` (linhas ~450 e ~464).

- [ ] **Passo 2: Escrever o teste que falha primeiro**

Adicione em `tests/test_neural_recommender.py`:

```python
def test_neural_recommender_respects_custom_num_negatives(sample_interactions):
    """NeuralRecommender deve propagar num_negatives customizado para o
    sampler de negativos, não usar o valor hardcoded de 4."""
    recommender = NeuralRecommender(
        embedding_dim=4,
        epochs=1,
        num_negatives=7,
    )
    recommender.fit(sample_interactions)

    assert recommender.num_negatives == 7
```

(Use a fixture `sample_interactions` já existente no arquivo de testes; se não existir, reaproveite a fixture usada por `test_neural_recommender_fit_and_recommend` ou equivalente já presente no arquivo.)

- [ ] **Passo 3: Rodar o teste para confirmar que falha**

```bash
pytest tests/test_neural_recommender.py::test_neural_recommender_respects_custom_num_negatives -v
```

Esperado: FAIL com `AttributeError: 'NeuralRecommender' object has no attribute 'num_negatives'` (o construtor ainda não aceita o parâmetro).

- [ ] **Passo 4: Adicionar `num_negatives` ao construtor de `NeuralRecommender`**

Em `src/recsys/recommenders/neural.py`, localize o `__init__` de `NeuralRecommender` e adicione o parâmetro, seguindo o padrão dos demais hiperparâmetros (ex.: `embedding_dim`, `dropout`):

```python
def __init__(
    self,
    embedding_dim: int = 8,
    mlp_layers: list[int] | None = None,
    dropout: float = 0.2,
    lr: float = 0.01,
    epochs: int = 20,
    batch_size: int = 1024,
    patience: int = 3,
    val_split: float = 0.02,
    num_negatives: int = 4,
    random_seed: int = 42,
) -> None:
    ...
    self.num_negatives = num_negatives
```

(Ajuste a lista de parâmetros conforme a assinatura real já existente — apenas insira `num_negatives: int = 4` na posição lógica e o `self.num_negatives = num_negatives` no corpo, sem remover nenhum parâmetro já presente.)

- [ ] **Passo 5: Propagar `self.num_negatives` para os pontos hardcoded**

Nas linhas ~450 e ~464 de `fit()`, troque o literal `4` por `self.num_negatives`:

```python
# Antes:
collate_fn = _NegativeSamplingCollate(
    num_items=self.num_items,
    num_negatives=4,
    ...
)
# Depois:
collate_fn = _NegativeSamplingCollate(
    num_items=self.num_items,
    num_negatives=self.num_negatives,
    ...
)
```

```python
# Antes (na amostragem de validação):
val_samples = _sample_negatives_dataset(
    val_df,
    num_items=self.num_items,
    num_negatives=4,
    ...
)
# Depois:
val_samples = _sample_negatives_dataset(
    val_df,
    num_items=self.num_items,
    num_negatives=self.num_negatives,
    ...
)
```

Os defaults hardcoded em `_NegativeSamplingCollate.__init__` (linha ~74), `_sample_negatives_dataset` (linha ~303) e `_sample_negatives` (linha ~376) podem permanecer como `num_negatives: int = 4` — são apenas valores-padrão de fallback para uso direto dessas funções fora do fluxo de `NeuralRecommender.fit()`. Não altere essas assinaturas.

- [ ] **Passo 6: Rodar o teste para confirmar que passa**

```bash
pytest tests/test_neural_recommender.py::test_neural_recommender_respects_custom_num_negatives -v
```

Esperado: PASS.

- [ ] **Passo 7: Rodar a suíte completa de testes do neural recommender**

```bash
pytest tests/test_neural_recommender.py -v
```

Esperado: todos os testes passam, incluindo os pré-existentes (nenhuma regressão).

- [ ] **Passo 8: Commit**

```bash
git add src/recsys/recommenders/neural.py tests/test_neural_recommender.py
git commit -m "fix: parametrize num_negatives in NeuralRecommender.fit"
```

---

### Tarefa 2: Eliminar risco de OOM em `recommend_batch` e tornar batch_size configurável

**Arquivos:**

- Modificar: `src/recsys/recommenders/neural.py:622,633`
- Teste: `tests/test_neural_recommender.py`

- [ ] **Passo 1: Ler o método `recommend_batch` completo**

```bash
grep -n "def recommend_batch" -A 40 src/recsys/recommenders/neural.py
```

Confirme a linha exata do `batch_size = 64` hardcoded e do `repeat_interleave(num_items)`.

- [ ] **Passo 2: Escrever o teste que falha primeiro**

```python
def test_recommend_batch_uses_configurable_inference_batch_size(sample_interactions):
    """recommend_batch deve aceitar um batch_size de inferência configurável
    em vez do valor hardcoded de 64, permitindo controlar o pico de memória."""
    recommender = NeuralRecommender(embedding_dim=4, epochs=1)
    recommender.fit(sample_interactions)

    user_ids = sample_interactions["user_id"].unique().tolist()
    results = recommender.recommend_batch(user_ids, top_k=5, inference_batch_size=2)

    assert set(results.keys()) == set(user_ids)
    assert all(len(v) <= 5 for v in results.values())
```

- [ ] **Passo 3: Rodar o teste para confirmar que falha**

```bash
pytest tests/test_neural_recommender.py::test_recommend_batch_uses_configurable_inference_batch_size -v
```

Esperado: FAIL com `TypeError: recommend_batch() got an unexpected keyword argument 'inference_batch_size'`.

- [ ] **Passo 4: Adicionar o parâmetro `inference_batch_size` e remover o `repeat_interleave` de larga escala**

Localize o corpo de `recommend_batch` (por volta da linha 610-640) e faça as seguintes alterações:

```python
# Assinatura antes:
def recommend_batch(
    self, user_ids: list[Any], top_k: int = 10
) -> dict[Any, list[str]]:
    ...
    batch_size = 64
    ...

# Assinatura depois:
def recommend_batch(
    self,
    user_ids: list[Any],
    top_k: int = 10,
    inference_batch_size: int = 64,
) -> dict[Any, list[str]]:
    ...
    batch_size = inference_batch_size
    ...
```

Em seguida, no trecho que hoje usa `repeat_interleave(num_items)` para expandir os usuários contra todo o catálogo de itens (~linha 633), a estratégia é processar os itens também em chunks dentro do loop de usuários, evitando alocar `len(batch_users) * num_items` de uma vez. Substitua:

```python
# Antes:
user_tensor = torch.tensor(u_indices, device=self.device).repeat_interleave(num_items)
item_tensor = torch.arange(num_items, device=self.device).repeat(len(u_indices))
scores = self.model(user_tensor, item_tensor).view(len(u_indices), num_items)
```

```python
# Depois:
item_chunk_size = min(num_items, 50_000)
scores = torch.empty((len(u_indices), num_items), device=self.device)
for item_start in range(0, num_items, item_chunk_size):
    item_end = min(item_start + item_chunk_size, num_items)
    chunk_len = item_end - item_start

    user_tensor = torch.tensor(u_indices, device=self.device).repeat_interleave(chunk_len)
    item_tensor = torch.arange(item_start, item_end, device=self.device).repeat(len(u_indices))

    scores[:, item_start:item_end] = self.model(user_tensor, item_tensor).view(
        len(u_indices), chunk_len
    )
```

Isso limita o pico de alocação a `len(batch_users) * item_chunk_size` em vez de `len(batch_users) * num_items`, resolvendo o risco de OOM para catálogos grandes sem alterar o resultado numérico.

- [ ] **Passo 5: Rodar o teste para confirmar que passa**

```bash
pytest tests/test_neural_recommender.py::test_recommend_batch_uses_configurable_inference_batch_size -v
```

Esperado: PASS.

- [ ] **Passo 6: Rodar o teste de paridade já existente (recommend_batch vs recommend individual)**

```bash
pytest tests/test_neural_recommender.py -k "batch" -v
```

Esperado: PASS — o chunking não deve alterar os scores computados, apenas a forma como são calculados.

- [ ] **Passo 7: Rodar a suíte completa**

```bash
pytest tests/test_neural_recommender.py -v
```

Esperado: todos os testes passam.

- [ ] **Passo 8: Commit**

```bash
git add src/recsys/recommenders/neural.py tests/test_neural_recommender.py
git commit -m "fix: chunk item scoring in recommend_batch to prevent OOM, expose inference_batch_size"
```

---

### Tarefa 3: Logar métricas via MLflow em `train.py`

**Arquivos:**

- Modificar: `src/recsys/pipeline/train.py`

- [ ] **Passo 1: Ler as funções `train_svd` e `train_neural` completas**

```bash
grep -n "def train_svd\|def train_neural\|mlflow.log" src/recsys/pipeline/train.py
```

Confirme onde `mlflow.log_param` já é chamado e onde inserir `mlflow.log_metric`.

- [ ] **Passo 2: Adicionar log de métricas em `train_svd`**

Após o treinamento do SVD (imediatamente antes de salvar/pickle o modelo), adicione:

```python
    recommender.fit(interactions)

    mlflow.log_metric("n_components", n_components)
    mlflow.log_metric("n_train_interactions", len(interactions))
```

(Ajuste os nomes de variáveis locais conforme o código real de `train_svd` — o objetivo é registrar ao menos o tamanho do dataset de treino, já disponível na função.)

- [ ] **Passo 3: Adicionar log de métricas em `train_neural`**

Localize o retorno do `recommender.fit(...)` no modo neural. O método `NeuralRecommender.fit` já registra métricas por época internamente (conforme mapeado no audit); adicione o log da métrica final de validação após o fit, usando o histórico de treino exposto pelo recommender:

```python
    recommender.fit(interactions)

    if recommender.best_val_loss is not None:
        mlflow.log_metric("best_val_loss", recommender.best_val_loss)
    mlflow.log_metric("epochs_trained", recommender.epochs_trained)
```

Se os atributos `best_val_loss` e `epochs_trained` não existirem em `NeuralRecommender`, use os nomes de atributos reais expostos pela classe (verifique com `grep -n "self\.\(best_val_loss\|epochs_trained\|history\)" src/recsys/recommenders/neural.py` antes de escrever esse trecho) — o objetivo é registrar ao menos uma métrica de resultado do treinamento neural no MLflow a partir de `train.py`, sem duplicar o log por época já feito internamente.

- [ ] **Passo 4: Rodar o teste de smoke do pipeline (se existir) ou executar manualmente**

```bash
pytest tests/ -k "train" -v
```

Se não houver teste dedicado a `train.py`, execute manualmente com um dataset pequeno de teste e confirme que não há erro:

```bash
python -m recsys.pipeline.train --input data/processed/train.parquet --output /tmp/model_test.pkl --mode baseline
```

Esperado: execução sem exceções, log de métricas visível no output do MLflow (ou nos arquivos em `mlruns/`).

- [ ] **Passo 5: Commit**

```bash
git add src/recsys/pipeline/train.py
git commit -m "feat: log training metrics to MLflow from train.py"
```

---

### Tarefa 4: Integrar `popularity.py` como terceira opção wired no DVC/params

**Arquivos:**

- Modificar: `params.yaml`
- Modificar: `src/recsys/pipeline/train.py`
- Ler (sem modificar): `src/recsys/recommenders/popularity.py`, `dvc.yaml`

- [ ] **Passo 1: Confirmar a assinatura de `PopularityRecommender`**

```bash
grep -n "class PopularityRecommender\|def __init__\|def fit\|def recommend" src/recsys/recommenders/popularity.py
```

- [ ] **Passo 2: Documentar a opção `popularity` em `params.yaml`**

Edite `params.yaml` para refletir as três opções válidas:

```yaml
model:
  recommender_type: neural  # baseline | neural | popularity
  embedding_dim: 8
  epochs: 20
  batch_size: 1024
  lr: 0.01
  dropout: 0.2
  patience: 3
  val_split: 0.02
```

- [ ] **Passo 3: Adicionar o modo `popularity` em `train.py`**

Localize o `argparse` com `choices=["baseline", "neural"]` e o dispatch em `main()`. Adicione a terceira opção:

```python
# Antes:
parser.add_argument(
    "--mode",
    choices=["baseline", "neural"],
    default=settings.model.recommender_type,
)

# Depois:
parser.add_argument(
    "--mode",
    choices=["baseline", "neural", "popularity"],
    default=settings.model.recommender_type,
)
```

Adicione a função de treino correspondente, seguindo o padrão de `train_svd`:

```python
def train_popularity(
    input_path: Path,
    output_path: Path,
    seed: int,
) -> None:
    """Treina o baseline de popularidade e salva o modelo.

    Args:
        input_path: Caminho do Parquet de treino.
        output_path: Caminho de saída do ``.pkl``.
        seed: Semente aleatória global.
    """
    from recsys.recommenders.popularity import PopularityRecommender

    fix_seeds(seed)

    interactions = pd.read_parquet(input_path)
    recommender = PopularityRecommender()
    recommender.fit(interactions)

    mlflow.log_param("mode", "popularity")
    mlflow.log_metric("n_train_interactions", len(interactions))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(recommender, f)
    mlflow.log_artifact(str(output_path))
```

E no dispatch de `main()`, adicione o branch correspondente:

```python
# Antes:
if args.mode == "baseline":
    train_svd(input_path, output_path, seed=settings.model.random_seed)
else:
    train_neural(...)

# Depois:
if args.mode == "baseline":
    train_svd(input_path, output_path, seed=settings.model.random_seed)
elif args.mode == "popularity":
    train_popularity(input_path, output_path, seed=settings.model.random_seed)
else:
    train_neural(...)
```

- [ ] **Passo 4: Escrever teste cobrindo o novo modo**

Adicione em `tests/test_recommenders.py` (ou crie teste equivalente se a estrutura de import de `train.py` exigir isso):

```python
def test_train_popularity_serializes_model(tmp_path, sample_interactions):
    from recsys.pipeline.train import train_popularity

    output_path = tmp_path / "popularity_model.pkl"
    train_popularity(input_path=None, output_path=output_path, seed=42)
```

Como `train_popularity` lê de `input_path` via `pd.read_parquet`, ajuste o teste para escrever um parquet temporário real:

```python
def test_train_popularity_serializes_model(tmp_path, sample_interactions):
    import pickle

    from recsys.pipeline.train import train_popularity
    from recsys.recommenders.popularity import PopularityRecommender

    input_path = tmp_path / "train.parquet"
    sample_interactions.to_parquet(input_path)
    output_path = tmp_path / "popularity_model.pkl"

    train_popularity(input_path=input_path, output_path=output_path, seed=42)

    assert output_path.exists()
    with open(output_path, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, PopularityRecommender)
```

(Reaproveite a fixture `sample_interactions` já usada nos demais testes do projeto.)

- [ ] **Passo 5: Rodar o teste para confirmar que passa**

```bash
pytest tests/test_recommenders.py::test_train_popularity_serializes_model -v
```

Esperado: PASS.

- [ ] **Passo 6: Rodar a suíte completa de testes**

```bash
pytest tests/ -v
```

Esperado: nenhuma regressão.

- [ ] **Passo 7: Commit**

```bash
git add params.yaml src/recsys/pipeline/train.py tests/test_recommenders.py
git commit -m "feat: wire PopularityRecommender as a selectable training mode"
```

---

### Tarefa 5: Adicionar validação de enum para `recommender_type` em `settings.py`

**Arquivos:**

- Modificar: `src/recsys/config/settings.py:53`
- Teste: `tests/test_settings.py`

- [ ] **Passo 1: Ler a classe `ModelSettings` completa**

```bash
grep -n "class ModelSettings" -A 10 src/recsys/config/settings.py
```

- [ ] **Passo 2: Escrever o teste que falha primeiro**

Em `tests/test_settings.py`:

```python
def test_recommender_type_rejects_invalid_value():
    """ModelSettings deve rejeitar valores de recommender_type fora do
    conjunto suportado (baseline, neural, popularity)."""
    import pytest
    from pydantic import ValidationError

    from recsys.config.settings import ModelSettings

    with pytest.raises(ValidationError):
        ModelSettings(recommender_type="invalid_type")



def test_recommender_type_accepts_valid_values():
    from recsys.config.settings import ModelSettings

    for valid in ("baseline", "neural", "popularity"):
        settings = ModelSettings(recommender_type=valid)
        assert settings.recommender_type == valid
```

- [ ] **Passo 3: Rodar o teste para confirmar que falha**

```bash
pytest tests/test_settings.py::test_recommender_type_rejects_invalid_value -v
```

Esperado: FAIL — nenhuma validação impede o valor `"invalid_type"` hoje.

- [ ] **Passo 4: Adicionar um `Literal` type para `recommender_type`**

Em `src/recsys/config/settings.py`, adicione o import de `Literal` e altere o campo:

```python
from typing import Literal
```

```python
# Antes:
recommender_type: str = Field(default="baseline", alias="RECOMMENDER_TYPE")

# Depois:
recommender_type: Literal["baseline", "neural", "popularity"] = Field(
    default="baseline", alias="RECOMMENDER_TYPE"
)
```

- [ ] **Passo 5: Rodar os testes para confirmar que passam**

```bash
pytest tests/test_settings.py -v
```

Esperado: `test_recommender_type_rejects_invalid_value` e `test_recommender_type_accepts_valid_values` passam, junto com todos os testes pré-existentes em `test_settings.py`.

- [ ] **Passo 6: Rodar a suíte completa**

```bash
pytest tests/ -v
```

Esperado: nenhuma regressão (o valor padrão `.env`/`params.yaml` deve continuar sendo `baseline` ou `neural`, ambos válidos no novo `Literal`).

- [ ] **Passo 7: Commit**

```bash
git add src/recsys/config/settings.py tests/test_settings.py
git commit -m "fix: constrain recommender_type to a Literal enum in ModelSettings"
```

---

## Verificação Final

- [ ] **Passo Final: Rodar a suíte completa e o pipeline DVC**

```bash
pytest tests/ -v
dvc repro
```

Esperado: todos os testes passam e o pipeline DVC executa as 5 stages (`preprocess`, `feature_eng`, `train`, `evaluate`, `run_tests`) sem erro, incluindo a nova opção `popularity` se selecionada em `params.yaml`.

- [ ] **Atualizar `exploration_audit_plan.md`**

Marque os 5 issues da tabela "Critical Issues to Fix Before Submission" como resolvidos, referenciando os commits desta implementação.