"""Interface abstrata para carregadores de datasets de interação.

Permite trocar a fonte de dados (arquivo local, HuggingFace, S3, etc.)
sem alterar o restante do pipeline. Implementações concretas serão
criadas na Etapa 4 (Pedro).

Implementações previstas:
    - AmazonReviewsLoader: carrega o subset Toys_and_Games via HuggingFace.
    - LocalParquetLoader: carrega interações de arquivo Parquet local.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseInteractionLoader(ABC):
    """Interface para carregadores de datasets de interação usuário-item.

    Define o contrato que qualquer fonte de dados deve satisfazer.
    O método ``load`` deve retornar um DataFrame normalizado com as
    colunas padrão do projeto: ``user_id``, ``item_id`` e ``rating``.

    Example:
        >>> class MyLoader(BaseInteractionLoader):
        ...     def load(self) -> pd.DataFrame:
        ...         return pd.DataFrame({
        ...             "user_id": ["u1"],
        ...             "item_id": ["i1"],
        ...             "rating": [5],
        ...         })
    """

    REQUIRED_COLUMNS: tuple[str, ...] = ("user_id", "item_id", "rating")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Carrega e retorna o dataset de interações.

        Returns:
            DataFrame com ao menos as colunas ``user_id``,
            ``item_id`` e ``rating``.
        """

    def validate(self, df: pd.DataFrame) -> None:
        """Verifica se o DataFrame possui as colunas obrigatórias.

        Args:
            df: DataFrame a ser validado.

        Raises:
            ValueError: Se alguma coluna obrigatória estiver ausente.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Colunas obrigatórias ausentes no dataset: {missing}. "
                f"Esperado: {list(self.REQUIRED_COLUMNS)}"
            )
