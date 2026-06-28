"""Script para registrar o modelo treinado no Model Registry do MLflow e promovê-lo a Production.

Este script carrega o modelo serializado de 'models/model.pkl', encapsula-o
em um wrapper customizado do MLflow PythonModel (para que possa realizar inferência
em lote com mappings de IDs integrados) e registra/promove o modelo no tracking server.

Usage:
    python scripts/register_model.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wrapper customizado para o modelo no MLflow
# ---------------------------------------------------------------------------


class RecommenderModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper para logar qualquer BaseRecommender do projeto no formato PythonModel do MLflow."""

    def load_context(self, context) -> None:
        """Carrega o modelo a partir do arquivo pkl/pth de contexto."""
        # Garante que o diretório src/ está no path do interpretador
        project_root = Path(__file__).resolve().parent.parent
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from recsys.pipeline.evaluate import load_model  # noqa: PLC0415

        model_file = Path(context.artifacts["model_file"])
        _log.info("Carregando modelo a partir do artefato do MLflow: %s", model_file)
        self.recommender = load_model(model_file)

    def predict(self, context, model_input) -> dict:
        """Gera recomendações em lote para os utilizadores de entrada.

        Args:
            context: Contexto carregado.
            model_input: DataFrame do Pandas contendo uma coluna 'user_id',
                ou uma lista/dicionário.

        Returns:
            Dicionário mapeando user_id para lista de recomendações top-K.
        """
        import pandas as pd  # noqa: PLC0415

        if isinstance(model_input, pd.DataFrame):
            if "user_id" in model_input.columns:
                user_ids = model_input["user_id"].tolist()
            else:
                user_ids = model_input.iloc[:, 0].tolist()
        else:
            user_ids = list(model_input)

        _log.info("Gerando recomendações em lote para %d usuários via MLflow predict...", len(user_ids))
        return self.recommender.recommend_batch(user_ids, top_k=10)


# ---------------------------------------------------------------------------
# Execução Principal
# ---------------------------------------------------------------------------


def main() -> None:
    model_path = Path("models/model.pkl")
    if not model_path.exists():
        _log.error("Arquivo de modelo '%s' não encontrado. Execute o pipeline primeiro.", model_path)
        sys.exit(1)

    # 1. Configurar URIs de Tracking
    tracking_uri = "http://localhost:5001"
    experiment_name = "neumf-instacart"
    model_name = "NeuMF-Instacart"

    _log.info("Conectando ao servidor do MLflow em '%s'...", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 2. Iniciar Run de Registro
    _log.info("Iniciando run de registro no MLflow...")
    with mlflow.start_run(run_name="registry_promotion") as run:
        # Registra parâmetros informais
        mlflow.log_param("registration_source", "scripts/register_model.py")

        # Loga o modelo encapsulado no wrapper customizado
        _log.info("Salvando e registrando modelo no Model Registry sob o nome '%s'...", model_name)
        model_info = mlflow.pyfunc.log_model(
            artifact_path="recommender_model",
            python_model=RecommenderModelWrapper(),
            artifacts={"model_file": str(model_path)},
            registered_model_name=model_name,
        )
        _log.info("Modelo registrado com sucesso. Run ID: %s", run.info.run_id)

    # 3. Transicionar o Modelo no Model Registry para Production
    _log.info("Iniciando transição de estágio no Model Registry...")
    client = MlflowClient()

    # Obtém a última versão cadastrada no Model Registry
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if not latest_versions:
        _log.error("Nenhuma versão do modelo '%s' encontrada no registry.", model_name)
        sys.exit(1)

    latest_version = latest_versions[-1].version
    _log.info("Última versão do modelo encontrada: %s", latest_version)

    # Transiciona a última versão do modelo para "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True,
    )
    _log.info(
        "=== SUCESSO: Versão %s do modelo '%s' promovida a 'Production'! ===",
        latest_version,
        model_name,
    )


if __name__ == "__main__":
    main()
