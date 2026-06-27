"""Etapa 3 do pipeline: treinamento do modelo de recomendação.

Suporta dois modos:
    - **baseline** (default): treina ``SVDRecommender`` (scikit-learn).
    - **neural**: treina ``NeuralRecommender`` (NeuMF/PyTorch) com early
      stopping e salva checkpoint ``.pth``.

O script está preparado para integração com MLflow — os comentários
``MLflow:`` indicam onde chamar ``log_param`` / ``log_metric``.

Usage:
    python -m recsys.pipeline.train \\
        --input data/processed/train.parquet \\
        --output models/model.pkl \\
        --mode baseline

    python -m recsys.pipeline.train \\
        --input data/processed/train.parquet \\
        --output models/neumf.pth \\
        --mode neural \\
        --embedding-dim 64 \\
        --epochs 100 \\
        --batch-size 256 \\
        --lr 0.001
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
import mlflow

from recsys.config import settings
from recsys.utils.seeds import fix_seeds

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def train_svd(
    input_path: Path,
    output_path: Path,
    seed: int,
    n_components: int = 50,
) -> None:
    """Treina o baseline SVD e salva o modelo.

    Args:
        input_path: Caminho do Parquet de treino.
        output_path: Caminho de saída do ``.pkl``.
        seed: Semente aleatória global.
        n_components: Número de componentes latentes do SVD.
    """
    from recsys.recommenders.baseline import SVDRecommender

    fix_seeds(seed)

    mlflow.log_param("model_type", "svd")
    mlflow.log_param("n_components", n_components)
    mlflow.log_param("seed", seed)

    _log.info("Lendo dados de treino de '%s'...", input_path)
    train_df: pd.DataFrame = pd.read_parquet(input_path)
    _log.info("%d interações de treino.", len(train_df))

    model = SVDRecommender(n_components=n_components, random_state=seed)
    model.fit(train_df)
    _log.info("Modelo SVD treinado.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)
    _log.info("Modelo salvo em '%s'.", output_path)

    mlflow.log_artifact(str(output_path))


def train_neural(
    input_path: Path,
    output_path: Path,
    seed: int,
    embedding_dim: int = 64,
    mlp_hidden_dims: list[int] | None = None,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 5,
    val_split: float = 0.2,
) -> None:
    """Treina o NeuMF (PyTorch) e salva checkpoint.

    Args:
        input_path: Caminho do Parquet de treino.
        output_path: Caminho de saída do ``.pth``.
        seed: Semente aleatória global.
        embedding_dim: Dimensão dos embeddings.
        mlp_hidden_dims: Dimensões ocultas da MLP.
        dropout: Taxa de dropout.
        epochs: Número máximo de épocas.
        batch_size: Tamanho do batch.
        lr: Learning rate.
        weight_decay: Regularização L2.
        patience: Paciência do early stopping.
        val_split: Proporção para validação.
    """
    from recsys.recommenders.neural import NeuralRecommender

    if mlp_hidden_dims is None:
        mlp_hidden_dims = [256, 128, 64]

    fix_seeds(seed)

    mlflow.log_param("model_type", "neumf")
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("mlp_hidden_dims", mlp_hidden_dims)
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lr", lr)
    mlflow.log_param("weight_decay", weight_decay)
    mlflow.log_param("patience", patience)
    mlflow.log_param("val_split", val_split)
    mlflow.log_param("seed", seed)

    _log.info("Lendo dados de treino de '%s'...", input_path)
    train_df: pd.DataFrame = pd.read_parquet(input_path)
    _log.info("%d interações de treino.", len(train_df))

    recommender = NeuralRecommender(
        embedding_dim=embedding_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        val_split=val_split,
        seed=seed,
    )
    recommender.fit(train_df)
    recommender.save(str(output_path))
    _log.info("Checkpoint NeuMF salvo em '%s'.", output_path)

    mlflow.log_artifact(str(output_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino do modelo de recomendação (baseline ou neural)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Parquet de treino.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/model.pkl"),
        help="Caminho de saída do modelo serializado.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória global.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "neural"],
        default=settings.model.recommender_type,
        help="Modo de treino: 'baseline' (SVD) ou 'neural' (NeuMF).",
    )

    # Parâmetros do modelo neural
    neural_group = parser.add_argument_group("neural")
    neural_group.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Dimensão dos embeddings (NeuMF).",
    )
    neural_group.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Dimensões ocultas da MLP (NeuMF).",
    )
    neural_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Taxa de dropout (NeuMF).",
    )
    neural_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número máximo de épocas (NeuMF).",
    )
    neural_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tamanho do batch (NeuMF).",
    )
    neural_group.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (NeuMF).",
    )
    neural_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Regularização L2 (NeuMF).",
    )
    neural_group.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciência do early stopping (NeuMF).",
    )
    neural_group.add_argument(
        "--val-split",
        type=float,
        default=0.02,
        help="Proporção para validação (NeuMF).",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada: delega para o modo de treino seleccionado."""
    args = _parse_args()

    if settings.mlflow.tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_param("execution_mode", args.mode)
        if args.mode == "baseline":
            train_svd(
                input_path=args.input,
                output_path=args.output,
                seed=args.seed,
            )
        elif args.mode == "neural":
            train_neural(
                input_path=args.input,
                output_path=args.output,
                seed=args.seed,
                embedding_dim=args.embedding_dim,
                mlp_hidden_dims=args.mlp_hidden_dims,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                val_split=args.val_split,
            )


if __name__ == "__main__":
    main()