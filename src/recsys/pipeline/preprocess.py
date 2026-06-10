"""Etapa 1 do pipeline: pré-processamento dos dados brutos.

Responsabilidades:
    1. Carregar os arquivos Instacart via ``InstacartLoader``.
    2. Remover nulos nas colunas obrigatórias.
    3. Aplicar filtro k-core (remover usuários/itens com poucas interações).
    4. Salvar o resultado em ``data/interim/preprocessed.parquet``.

Usage:
    python -m recsys.pipeline.preprocess \\
        --raw-dir data/raw \\
        --output data/interim/preprocessed.parquet \\
        --kcore 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from recsys.data.instacart_loader import InstacartLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K-core filtering
# ---------------------------------------------------------------------------


def apply_kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Remove iterativamente usuários e itens com menos de ``k`` interações.

    Args:
        df: DataFrame com colunas ``user_id``, ``item_id`` e ``rating``.
        k: Número mínimo de interações para manter um usuário ou item.

    Returns:
        DataFrame filtrado, potencialmente menor que o original.
    """
    while True:
        before = len(df)

        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]

        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= k].index)]

        if len(df) == before:
            break

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def preprocess(raw_dir: Path, output: Path, kcore: int) -> None:
    """Executa o pré-processamento completo.

    Args:
        raw_dir: Diretório com os arquivos brutos do Instacart.
        output: Caminho do arquivo Parquet de saída.
        kcore: Threshold mínimo de interações para k-core filtering.
    """
    _log.info("Carregando dataset Instacart de '%s'...", raw_dir)
    loader = InstacartLoader(raw_dir)
    df = loader.load()
    _log.info("Dataset carregado: %d interações brutas.", len(df))

    df = df.dropna(subset=["user_id", "item_id", "rating"])
    _log.info("Após remoção de nulos: %d interações.", len(df))

    if kcore > 1:
        df = apply_kcore(df, kcore)
        _log.info("Após k-core (k=%d): %d interações.", kcore, len(df))

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    _log.info("Salvo em '%s'.", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pré-processamento Instacart")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Diretório com os CSVs brutos do Instacart.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/preprocessed.parquet"),
        help="Caminho do Parquet de saída.",
    )
    parser.add_argument(
        "--kcore",
        type=int,
        default=5,
        help="Número mínimo de interações por usuário e item.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(raw_dir=args.raw_dir, output=args.output, kcore=args.kcore)