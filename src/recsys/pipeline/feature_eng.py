"""Etapa 2 do pipeline: feature engineering.

Responsabilidades:
    1. Ler o dataset pré-processado.
    2. Mapear ``user_id`` e ``item_id`` para índices inteiros contíguos
       (``user_idx``, ``item_idx``) — requisito padrão de embeddings em RecSys.
    3. Dividir em treino (80%) e teste (20%) com semente fixa.
    4. Salvar: ``features.parquet`` (completo), ``train.parquet`` e ``test.parquet``.

Usage:
    python -m recsys.pipeline.feature_eng \\
        --input data/interim/preprocessed.parquet \\
        --output-dir data/processed \\
        --test-size 0.2 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def encode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas ``user_idx`` e ``item_idx`` com índices inteiros.

    A codificação é determinística: IDs são ordenados antes de mapear,
    garantindo reprodutibilidade independente de ordem de leitura.

    Args:
        df: DataFrame com colunas ``user_id`` e ``item_id``.

    Returns:
        DataFrame com as colunas originais mais ``user_idx`` e ``item_idx``.
    """
    user_map = {uid: idx for idx, uid in enumerate(sorted(df["user_id"].unique()))}
    item_map = {iid: idx for idx, iid in enumerate(sorted(df["item_id"].unique()))}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map).astype("int32")
    df["item_idx"] = df["item_id"].map(item_map).astype("int32")
    return df


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def feature_eng(
    input_path: Path,
    output_dir: Path,
    test_size: float,
    seed: int,
) -> None:
    """Executa a engenharia de features completa.

    Args:
        input_path: Caminho do Parquet pré-processado.
        output_dir: Diretório de destino dos Parquets de saída.
        test_size: Proporção do conjunto de teste (ex: 0.2 = 20%).
        seed: Semente aleatória para reprodutibilidade.
    """
    _log.info("Lendo '%s'...", input_path)
    df = pd.read_parquet(input_path)
    _log.info("%d interações lidas.", len(df))

    df = encode_ids(df)
    _log.info(
        "IDs codificados: %d usuários, %d itens.",
        df["user_idx"].nunique(),
        df["item_idx"].nunique(),
    )

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )
    _log.info("Split: %d treino / %d teste.", len(train_df), len(test_df))

    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_dir / "features.parquet", index=False)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    _log.info("Arquivos salvos em '%s'.", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering Instacart")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/preprocessed.parquet"),
        help="Parquet pré-processado de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Diretório de saída.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de teste.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    feature_eng(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
    )