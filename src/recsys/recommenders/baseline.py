"""Recomendador baseline usando TruncatedSVD do scikit-learn.

Implementa filtro colaborativo como baseline de comparação.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from recsys.recommenders.base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """Filtro colaborativo com TruncatedSVD (scikit-learn).

    Aplica SVD na matriz usuário-item para aprender representações
    latentes, servindo de baseline para modelos neurais.
    """

    def __init__(self, n_components: int = 50, random_state: int = 42) -> None:
        """Inicializa o recomendador SVD.

        Args:
            n_components: Número de dimensões latentes.
            random_state: Seed para reprodutibilidade.
        """
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._user_idx: dict[Any, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o SVD na matriz esparsa usuário-item.

        Args:
            interactions: DataFrame com user_id, item_id e rating.
        """
        matrix, users, items = self._create_sparse_matrix(interactions)

        self._user_idx = {u: i for i, u in enumerate(users)}
        self._idx_to_item = {i: str(i_id) for i, i_id in enumerate(items)}

        self._user_factors = self._svd.fit_transform(matrix)
        self._item_factors = self._svd.components_
        self._is_fitted = True

    def _create_sparse_matrix(
        self, interactions: pd.DataFrame
    ) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Cria matriz esparsa usuário-item.

        Args:
            interactions: DataFrame de interações.

        Returns:
            Matriz esparsa, array de usuários e array de itens.
        """
        users = interactions["user_id"].unique()
        items = interactions["item_id"].unique()

        u_map = {u: i for i, u in enumerate(users)}
        i_map = {i_id: i for i, i_id in enumerate(items)}

        row = interactions["user_id"].map(u_map).to_numpy()
        col = interactions["item_id"].map(i_map).to_numpy()
        data = interactions["rating"].to_numpy()

        matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
        return matrix, users, items

    def recommend(self, user_id: Any, top_k: int = 10) -> list[str]:
        """Retorna recomendações top-K para o usuário.

        Args:
            user_id: ID do usuário alvo.
            top_k: Número máximo de itens a recomendar.

        Returns:
            Lista de item_ids recomendados.

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if not self._is_fitted:
            raise RuntimeError("Chame fit() antes de recommend().")

        if user_id not in self._user_idx:
            return []

        u_idx = self._user_idx[user_id]
        scores = np.dot(self._user_factors[u_idx], self._item_factors)  # type: ignore

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._idx_to_item[idx] for idx in top_indices]
