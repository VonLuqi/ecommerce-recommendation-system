"""Módulo base para estratégias de recomendação.

Define a interface abstrata (Strategy) que todas as implementações
concretas de recomendadores devem seguir. Novas abordagens — como
NeuMF ou filtros colaborativos — implementam esta interface sem
alterar o código cliente.

Princípios SOLID aplicados:
    - SRP: cada estratégia tem uma única responsabilidade.
    - OCP: novas estratégias são adicionadas sem modificar a base.
    - LSP: qualquer concreto pode substituir BaseRecommender.
    - ISP: interface enxuta com apenas os métodos necessários.
    - DIP: clientes dependem da abstração, não de implementações.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseRecommender(ABC):
    """Interface Strategy para recomendadores de produtos.

    Todas as implementações concretas (baseline de popularidade,
    NeuMF, filtro colaborativo, etc.) devem herdar desta classe e
    implementar os métodos ``fit`` e ``recommend``.

    Example:
        >>> class MyRecommender(BaseRecommender):
        ...     def fit(self, interactions: pd.DataFrame) -> None:
        ...         pass
        ...     def recommend(self, user_id: Any, top_k: int = 10) -> list[str]:
        ...         return []
    """

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o recomendador com os dados de interação usuário-item.

        Args:
            interactions: DataFrame com ao menos as colunas
                ``user_id``, ``item_id`` e ``rating``.
        """

    @abstractmethod
    def recommend(self, user_id: Any, top_k: int = 10) -> list[str]:
        """Retorna os top-K itens recomendados para um usuário.

        Args:
            user_id: Identificador do usuário-alvo.
            top_k: Número de recomendações a retornar. Padrão: 10.

        Returns:
            Lista de identificadores de itens ordenada por relevância
            predita, do mais ao menos relevante.
        """

    def recommend_batch(
        self,
        user_ids: list[Any],
        top_k: int = 10,
    ) -> dict[Any, list[str]]:
        """Retorna as recomendações top-K em lote para vários usuários.

        Implementação padrão executa recommend() em loop. Pode ser sobrescrita
        por subclasses para melhoria de desempenho através de vetorização.

        Args:
            user_ids: Lista de identificadores de usuários.
            top_k: Número de recomendações a retornar. Padrão: 10.

        Returns:
            Dicionário mapeando cada user_id para sua lista de recomendações.
        """
        return {u_id: self.recommend(u_id, top_k) for u_id in user_ids}
