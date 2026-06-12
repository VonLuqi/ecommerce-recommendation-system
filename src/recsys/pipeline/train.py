"""Etapa 3 do pipeline: treinamento do modelo baseline.

Treina o ``PopularityRecommender`` — baseline de popularidade global —
sobre os dados de treino e serializa o modelo com ``pickle``.

Este baseline serve como piso de performance que o NeuMF (Etapa 7)
precisa superar.

Usage:
    python -m recsys.pipeline.train \\
        --input data/processed/train.parquet \\
        --output models/model.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from recsys.recommenders.baseline import SVDRecommender
from recsys.utils.seeds import fix_seeds

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def train(input_path: Path, output_path: Path, seed: int) -> None:
    """Treina o baseline SVD e salva o modelo.

    Args:
        input_path: Caminho do Parquet de treino.
        output_path: Caminho de saída do arquivo ``.pkl``.
        seed: Semente aleatória (aplicada globalmente para reprodutibilidade).
    """
    fix_seeds(seed)

    _log.info("Lendo dados de treino de '%s'...", input_path)
    train_df: pd.DataFrame = pd.read_parquet(input_path)
    _log.info("%d interações de treino.", len(train_df))

    model = SVDRecommender(n_components=50, random_state=seed)
    model.fit(train_df)
    _log.info("Modelo SVD treinado.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)
    _log.info("Modelo salvo em '%s'.", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino do modelo baseline")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(input_path=args.input, output_path=args.output, seed=args.seed)