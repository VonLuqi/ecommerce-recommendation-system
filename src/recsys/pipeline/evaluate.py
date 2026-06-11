"""Etapa 4 do pipeline: avaliação do modelo.

Métricas computadas:
    - **HR@K** (Hit Rate at K): proporção de interações de teste em que
      o item real aparece no top-K recomendado.
    - **Coverage@K**: proporção do catálogo **completo de treino** coberta
      pelas recomendações (usa ``--train-data`` para definir o catálogo).
    - **num_users_evaluated**: quantidade de usuários no conjunto de teste.
    - **num_interactions_evaluated**: total de pares (user, item) avaliados.

Os resultados são persistidos em ``metrics.json`` para rastreio pelo DVC
e futuramente pelo MLflow (Etapa 8).

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

import pandas as pd

from recsys.recommenders.base import BaseRecommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def hit_rate_at_k(
    model: BaseRecommender,
    test_df: pd.DataFrame,
    top_k: int,
) -> float:
    """Calcula HR@K para todas as interações no conjunto de teste.

    Para cada par (user_id, item_id) no teste, verifica se ``item_id``
    está entre os top-K retornados pelo modelo para ``user_id``.

    Args:
        model: Recomendador já treinado (``fit`` já chamado).
        test_df: DataFrame de teste com colunas ``user_id`` e ``item_id``.
        top_k: Número de recomendações a considerar.

    Returns:
        HR@K como float entre 0.0 e 1.0.
    """
    hits = 0
    total = len(test_df)

    # Pré-computar recomendações por usuário evita chamadas repetidas
    recommendations: dict[str, set[str]] = {}
    for user_id in test_df["user_id"].unique():
        recs = model.recommend(user_id=user_id, top_k=top_k)
        recommendations[user_id] = set(recs)

    for _, row in test_df.iterrows():
        if row["item_id"] in recommendations.get(str(row["user_id"]), set()):
            hits += 1

    return hits / total if total > 0 else 0.0


def catalog_coverage_at_k(
    model: BaseRecommender,
    test_df: pd.DataFrame,
    full_catalog: set[str],
    top_k: int,
) -> float:
    """Calcula a cobertura do catálogo pelas recomendações top-K.

    Args:
        model: Recomendador já treinado.
        test_df: DataFrame de teste (para obter os usuários únicos).
        full_catalog: Conjunto de todos os ``item_id`` do catálogo.
        top_k: Número de recomendações a considerar por usuário.

    Returns:
        Proporção do catálogo coberta, entre 0.0 e 1.0.
    """
    recommended: set[str] = set()
    for user_id in test_df["user_id"].unique():
        recs = model.recommend(user_id=user_id, top_k=top_k)
        recommended.update(recs)

    return len(recommended & full_catalog) / len(full_catalog) if full_catalog else 0.0


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def evaluate(
    model_path: Path,
    data_path: Path,
    train_path: Path,
    output_path: Path,
    top_k: int,
) -> None:
    """Carrega o modelo, avalia no conjunto de teste e salva métricas.

    Args:
        model_path: Caminho do arquivo ``.pkl`` do modelo treinado.
        data_path: Caminho do Parquet de teste.
        train_path: Caminho do Parquet de treino (para definir o catálogo completo).
        output_path: Caminho do ``metrics.json`` de saída.
        top_k: Número de recomendações para HR@K e Coverage@K.
    """
    _log.info("Carregando modelo de '%s'...", model_path)
    with model_path.open("rb") as f:
        model: BaseRecommender = pickle.load(f)  # noqa: S301

    _log.info("Lendo conjunto de teste de '%s'...", data_path)
    test_df: pd.DataFrame = pd.read_parquet(data_path)
    _log.info("%d interações de teste, %d usuários únicos.",
              len(test_df), test_df["user_id"].nunique())

    _log.info("Lendo catálogo completo de '%s'...", train_path)
    train_df: pd.DataFrame = pd.read_parquet(train_path, columns=["item_id"])
    full_catalog: set[str] = set(train_df["item_id"].astype(str).unique())
    _log.info("Catálogo completo: %d itens únicos de treino.", len(full_catalog))

    _log.info("Calculando HR@%d...", top_k)
    hr = hit_rate_at_k(model, test_df, top_k)

    _log.info("Calculando Coverage@%d...", top_k)
    coverage = catalog_coverage_at_k(model, test_df, full_catalog, top_k)

    metrics = {
        f"hr_at_{top_k}": round(hr, 6),
        f"coverage_at_{top_k}": round(coverage, 6),
        "num_users_evaluated": int(test_df["user_id"].nunique()),
        "num_interactions_evaluated": int(len(test_df)),
        "top_k": top_k,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))

    _log.info("Métricas salvas em '%s':", output_path)
    for key, value in metrics.items():
        _log.info("  %s: %s", key, value)


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
        help="Parquet de treino (usado para definir o catálogo completo).",
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
        help="Número de recomendações para HR@K e Coverage@K.",
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