"""Etapa 4 do pipeline: avaliação do modelo.

Métricas computadas:
    - Precision@K
    - Recall@K
    - NDCG@K
    - MAP@K

Os resultados são persistidos em ``metrics.json`` para rastreio pelo DVC.

Usage:
    python -m recsys.pipeline.evaluate \\
        --model models/model.pkl \\
        --data data/processed/test.parquet \\
        --train-data data/processed/train.parquet \\
        --output metrics.json \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config import settings
from recsys.metrics.evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recsys.recommenders.base import BaseRecommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def calculate_metrics(
    model: BaseRecommender,
    test_df: pd.DataFrame,
    top_k: int,
) -> dict[str, float]:
    """Calcula as métricas para todos os usuários no conjunto de teste usando processamento em lote.

    Args:
        model: Recomendador já treinado.
        test_df: DataFrame de teste com colunas ``user_id`` e ``item_id``.
        top_k: Número de recomendações a considerar.

    Returns:
        Dicionário com médias das métricas calculadas.
    """
    precisions = []
    recalls = []
    ndcgs = []
    maps = []

    user_true_items = test_df.groupby("user_id")["item_id"].apply(list).to_dict()
    user_ids = list(user_true_items.keys())

    _log.info("Gerando recomendações em lote para %d usuários...", len(user_ids))
    all_recommendations = model.recommend_batch(user_ids, top_k=top_k)

    for user_id, y_true in user_true_items.items():
        y_true_str = [str(item) for item in y_true]
        y_pred = all_recommendations.get(user_id, [])

        precisions.append(precision_at_k(y_true_str, y_pred, top_k))
        recalls.append(recall_at_k(y_true_str, y_pred, top_k))
        ndcgs.append(ndcg_at_k(y_true_str, y_pred, top_k))
        maps.append(map_at_k(y_true_str, y_pred, top_k))

    return {
        f"precision_at_{top_k}": sum(precisions) / len(precisions)
        if precisions
        else 0.0,
        f"recall_at_{top_k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"ndcg_at_{top_k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        f"map_at_{top_k}": sum(maps) / len(maps) if maps else 0.0,
    }


def load_model(model_path: Path) -> BaseRecommender:
    """Carrega o modelo de recomendação suportando pickle (.pkl) e PyTorch (.pth) dinamicamente.

    Args:
        model_path: Caminho do modelo serializado.

    Returns:
        O recomendador carregado.
    """
    try:
        with model_path.open("rb") as f:
            return pickle.load(f)  # noqa: S301
    except Exception:
        _log.info(
            "Falha ao carregar com pickle. Tentando carregar como modelo PyTorch (NeuMF)..."
        )
        from recsys.recommenders.neural import NeuralRecommender

        recommender = NeuralRecommender()
        recommender.load(model_path)
        return recommender


def evaluate(
    model_path: Path,
    data_path: Path,
    train_path: Path,
    output_path: Path,
    top_k: int,
) -> None:
    """Carrega o modelo, avalia no conjunto de teste e salva métricas.

    Args:
        model_path: Caminho do arquivo ``.pkl`` ou ``.pth`` do modelo treinado.
        data_path: Caminho do Parquet de teste.
        train_path: Caminho do Parquet de treino (mantido para CLI).
        output_path: Caminho do ``metrics.json`` de saída.
        top_k: Número de recomendações para as métricas.
    """
    _log.info("Carregando modelo de '%s'...", model_path)
    model = load_model(model_path)

    _log.info("Lendo conjunto de teste de '%s'...", data_path)
    test_df: pd.DataFrame = pd.read_parquet(data_path)

    num_users = test_df["user_id"].nunique()
    _log.info("%d interações de teste, %d usuários únicos.", len(test_df), num_users)

    _log.info("Calculando métricas @%d...", top_k)
    metrics = calculate_metrics(model, test_df, top_k)

    metrics["num_users_evaluated"] = int(num_users)
    metrics["num_interactions_evaluated"] = int(len(test_df))
    metrics["top_k"] = top_k

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))

    _log.info("Métricas salvas em '%s':", output_path)
    for key, value in metrics.items():
        _log.info("  %s: %s", key, value)

    try:
        if settings.mlflow.tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

        if mlflow.active_run():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
        else:
            mlflow.set_experiment(settings.mlflow.experiment_name)
            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_param("evaluate_only", True)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, float(value))
    except Exception as e:
        _log.warning("Falha ao logar no MLflow: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação do modelo de recomendação")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/model.pkl"),
        help="Caminho do modelo serializado.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/test.parquet"),
        help="Parquet de teste.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Parquet de treino (mantido para não quebrar dvc.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics.json"),
        help="Arquivo JSON de saída das métricas.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Número de recomendações para as métricas.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        model_path=args.model,
        data_path=args.data,
        train_path=args.train_data,
        output_path=args.output,
        top_k=args.top_k,
    )
