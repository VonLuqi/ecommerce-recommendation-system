"""Carregador concreto para o dataset Instacart Market Basket.

O Instacart não possui ratings explícitos. A estratégia adotada é tratar
a **frequência de compra** (quantas vezes um par user-produto aparece no
histórico ``order_products__prior``) como rating implícito.

Arquivos esperados em ``raw_dir``:
    - ``orders.csv``
    - ``order_products__prior.csv``

Download: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.data.loader import BaseInteractionLoader

_ORDERS_FILE = "orders.csv"
_ORDER_PRODUCTS_FILE = "order_products__prior.csv"


class InstacartLoader(BaseInteractionLoader):
    """Carrega e normaliza o dataset Instacart Market Basket.

    Lê os arquivos ``orders.csv`` e ``order_products__prior.csv``,
    une-os e agrega por (user_id, item_id) usando frequência de
    compra como rating implícito.

    Args:
        raw_dir: Diretório contendo os arquivos CSV brutos do Instacart.

    Raises:
        FileNotFoundError: Se algum dos arquivos CSV esperados não existir.

    Example:
        >>> loader = InstacartLoader("data/raw")
        >>> df = loader.load()
        >>> df.columns.tolist()
        ['user_id', 'item_id', 'rating']
        >>> df.dtypes["rating"]
        dtype('float64')
    """

    def __init__(self, raw_dir: Path | str) -> None:
        self._raw_dir = Path(raw_dir)

    def load(self) -> pd.DataFrame:
        """Carrega, une e agrega os arquivos Instacart.

        Returns:
            DataFrame com as colunas ``user_id`` (str), ``item_id`` (str)
            e ``rating`` (float) representando a frequência de compra.

        Raises:
            FileNotFoundError: Se algum arquivo CSV não for encontrado.
            ValueError: Se alguma coluna obrigatória estiver ausente após
                o processamento.
        """
        orders = self._read_csv(_ORDERS_FILE, usecols=["order_id", "user_id"])
        order_products = self._read_csv(
            _ORDER_PRODUCTS_FILE, usecols=["order_id", "product_id"]
        )

        df = self._merge_and_aggregate(orders, order_products)
        self.validate(df)
        return df

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _read_csv(self, filename: str, usecols: list[str]) -> pd.DataFrame:
        """Lê um CSV do diretório raw e retorna apenas as colunas solicitadas.

        Args:
            filename: Nome do arquivo CSV dentro de ``raw_dir``.
            usecols: Colunas a carregar.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        path = self._raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path.resolve()}\n"
                "Baixe o dataset em: "
                "https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis"
            )
        return pd.read_csv(path, usecols=usecols)

    def _merge_and_aggregate(
        self,
        orders: pd.DataFrame,
        order_products: pd.DataFrame,
    ) -> pd.DataFrame:
        """Une pedidos com produtos e agrega frequência de compra.

        Args:
            orders: DataFrame com ``order_id`` e ``user_id``.
            order_products: DataFrame com ``order_id`` e ``product_id``.

        Returns:
            DataFrame normalizado com colunas ``user_id``, ``item_id``
            e ``rating`` (frequência de compra, mínimo 1.0).
        """
        merged = order_products.merge(orders, on="order_id", how="inner")

        aggregated = (
            merged.groupby(["user_id", "product_id"], sort=False)
            .size()
            .reset_index(name="rating")
        )

        aggregated = aggregated.rename(columns={"product_id": "item_id"})
        aggregated["user_id"] = aggregated["user_id"].astype(str)
        aggregated["item_id"] = aggregated["item_id"].astype(str)
        aggregated["rating"] = aggregated["rating"].astype(float)

        return aggregated[["user_id", "item_id", "rating"]]
