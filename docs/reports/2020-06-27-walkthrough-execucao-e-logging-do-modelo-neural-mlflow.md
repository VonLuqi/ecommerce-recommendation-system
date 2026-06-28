# Walkthrough — Execução e Registro do Modelo de Recomendação

Este documento resume as modificações, execuções do pipeline DVC localmente e a promoção final do modelo no Model Registry do MLflow.

---

## 1. Modificações Efetuadas

### 🛠️ Correção de Bugs de Dispositivo e Tipo (MPS / Mac)
*   **Problema**: O treinamento e teste em dispositivos Apple Silicon (GPU MPS) quebravam com `TypeError: Cannot convert a MPS Tensor to float64 dtype`.
*   **Solução**: No arquivo [neural.py](file:///Users/eduardobatista/Code/ecommerce-recommendation-system/src/recsys/recommenders/neural.py#L753), realizamos o cast explícito de ratings para `torch.float32` antes de enviar o tensor para o dispositivo, permitindo a aceleração via hardware no Mac com 100% de sucesso nos testes.

### ⚡ Otimização do Tempo de Inferência e Avaliação (NeuMF)
*   **Problema**: A inferência em lote (`recommend_batch`) expandia tensores de entrada em CPU usando NumPy e copiava grandes matrizes cartesiana para a GPU MPS a cada lote de 16 usuários. Isso gerava uma cópia acumulada de mais de **147 GB** de memória e sobrecarregava o alocador de Metal do macOS.
*   **Solução**: No arquivo [neural.py](file:///Users/eduardobatista/Code/ecommerce-recommendation-system/src/recsys/recommenders/neural.py#L616-L620), vetorizamos a expansão de tensores diretamente no dispositivo (`mps`) usando `repeat_interleave` e `repeat`. Também aumentamos o tamanho do lote de inferência de `16` para `64`.
*   **Resultado**: O tempo da etapa `evaluate` caiu drasticamente de mais de 15 minutos para poucos minutos, permitindo a execução rápida e viável do pipeline.

### 🌐 Configuração de Proxy de Artefatos no MLflow
*   **Problema**: Rodando o pipeline no host local do macOS e o MLflow em um container Docker separado impedia o cliente local de salvar arquivos diretamente na pasta interna `/app/mlruns` do container.
*   **Solução**: Em [docker-compose.yml](file:///Users/eduardobatista/Code/ecommerce-recommendation-system/docker-compose.yml), adicionamos as flags `--serve-artifacts` e `--default-artifact-root mlflow-artifacts:/` para expor uma API HTTP de uploads, permitindo o logging remoto direto a partir do host.

---

## 2. Resultados das Runs Executadas

As 3 runs planejadas foram executadas localmente e registradas no servidor do MLflow (`http://localhost:5001`):

| Métrica @10 | Run 1: Baseline SVD | Run 2: NeuMF Smoke-Test (1 epoch) | Run 3: NeuMF Final (20 epochs) |
| :--- | :---: | :---: | :---: |
| **NDCG@10** | `0.0217` (2.17%) | `0.0680` (6.80%) | **`0.0710` (7.10%)** |
| **MAP@10** | `0.0073` (0.73%) | `0.0283` (2.83%) | **`0.0296` (2.96%)** |
| **Precision@10** | `0.0202` (2.02%) | `0.0540` (5.40%) | **`0.0563` (5.63%)** |
| **Recall@10** | `0.0259` (2.59%) | `0.0490` (4.90%) | **`0.0524` (5.24%)** |
| **Run Name** | `classy-bee-362` | `smiling-dolphin-923` | `learned-asp-680` |

> [!NOTE]
> O modelo **NeuMF Final** com 20 épocas superou o Baseline SVD em mais de **3.2x** em NDCG@10 (7.10% vs 2.17%), provando o acerto na arquitetura e a qualidade do treinamento acelerado por GPU.

---

## 3. Registro e Promoção do Modelo

Criamos o script automatizado [register_model.py](file:///Users/eduardobatista/Code/ecommerce-recommendation-system/scripts/register_model.py) para registrar o modelo e promovê-lo no Model Registry:
1.  **Wrapper Customizado**: Criamos `RecommenderModelWrapper` derivado de `mlflow.pyfunc.PythonModel` para encapsular a carga e predict de forma independente.
2.  **Exportação do Ambiente**: O MLflow utilizou `uv export` para gerar e empacotar o arquivo de requerimentos `requirements.txt` junto aos arquivos de configuração `pyproject.toml` e `uv.lock`.
3.  **Promoção para Production**: O script contatou o Model Registry e promoveu a versão 1 do modelo `NeuMF-Instacart` ao estágio **Production** com sucesso.

### 📝 Verificação no Model Registry:
```bash
$ uv run python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5001'); from mlflow.tracking import MlflowClient; client = MlflowClient(); print([(mv.name, mv.version, mv.current_stage) for mv in client.get_latest_versions('NeuMF-Instacart')])"
[('NeuMF-Instacart', '1', 'Production')]
```

---

## 4. Testes Unitários

Todos os 40 testes do projeto continuam passando com sucesso:
```bash
============================== 40 passed in 3.01s ==============================
```
