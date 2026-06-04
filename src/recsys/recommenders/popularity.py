"""Recomendador baseline baseado em popularidade global.

Estratégia concreta mais simples possível: recomendar os itens com
maior número de interações, ignorando o perfil individual do usuário.

Serve como floor de performance — qualquer modelo de ML deve superar
este baseline para justificar sua complexidade adicional.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from recsys.recommenders.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recomenda os itens globalmente mais populares como baseline.

    Não personaliza recomendações por usuário. Utilizado na Etapa 6
    para estabelecer o piso mínimo de performance (HR@10, NDCG@10)
    que o NeuMF deve superar.

    Attributes:
        _popular_items: Lista de item_ids ordenada por frequência
            decrescente de interação. Preenchida após ``fit``.

    Example:
        >>> import pandas as pd
        >>> rec = PopularityRecommender()
        >>> interactions = pd.DataFrame({
        ...     "user_id": ["u1", "u1", "u2"],
        ...     "item_id": ["i1", "i2", "i1"],
        ...     "rating": [5, 4, 3],
        ... })
        >>> rec.fit(interactions)
        >>> rec.recommend(user_id="u1", top_k=1)
        ['i1']
    """

    def __init__(self) -> None:
        self._popular_items: list[str] = []

    def fit(self, interactions: pd.DataFrame) -> None:
        """Ordena itens por frequência de interação decrescente.

        Args:
            interactions: DataFrame com ao menos a coluna ``item_id``.

        Raises:
            KeyError: Se a coluna ``item_id`` não estiver presente.
        """
        counts: Counter[str] = Counter(interactions["item_id"])
        self._popular_items = [item for item, _ in counts.most_common()]

    def recommend(self, user_id: Any, top_k: int = 10) -> list[str]:
        """Retorna os top-K itens mais populares, sem personalização.

        Args:
            user_id: Ignorado — popularidade é independente do usuário.
            top_k: Número de recomendações a retornar. Padrão: 10.

        Returns:
            Lista dos ``top_k`` item_ids mais frequentes no treino.

        Raises:
            RuntimeError: Se ``fit`` não tiver sido chamado antes.
        """
        if not self._popular_items:
            raise RuntimeError(
                "O recomendador precisa ser treinado antes de recomendar. "
                "Chame fit() com os dados de interação."
            )
        return self._popular_items[:top_k]
