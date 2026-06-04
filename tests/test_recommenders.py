"""Testes para as estratégias de recomendação.

Cobre a interface BaseRecommender e a implementação concreta
PopularityRecommender.
"""

import pandas as pd
import pytest

from recsys.recommenders.base import BaseRecommender
from recsys.recommenders.popularity import PopularityRecommender

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_interactions() -> pd.DataFrame:
    """DataFrame mínimo de interações para os testes."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i2", "i1", "i3", "i1"],
            "rating": [5, 4, 3, 5, 4],
        }
    )


# ---------------------------------------------------------------------------
# BaseRecommender — contrato da interface
# ---------------------------------------------------------------------------


def test_base_recommender_is_abstract() -> None:
    """BaseRecommender não pode ser instanciada diretamente."""
    with pytest.raises(TypeError):
        BaseRecommender()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# PopularityRecommender
# ---------------------------------------------------------------------------


class TestPopularityRecommender:
    """Testes unitários do recomendador de popularidade."""

    def test_recommend_raises_before_fit(self) -> None:
        """recommend() deve lançar RuntimeError se fit() não foi chamado."""
        rec = PopularityRecommender()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            rec.recommend(user_id="u1")

    def test_fit_and_recommend_top_k(self, sample_interactions: pd.DataFrame) -> None:
        """O item mais popular deve aparecer em primeiro lugar."""
        rec = PopularityRecommender()
        rec.fit(sample_interactions)
        result = rec.recommend(user_id="u1", top_k=1)

        assert result == ["i1"], "i1 aparece 3 vezes e deve ser o mais popular"

    def test_recommend_respects_top_k(self, sample_interactions: pd.DataFrame) -> None:
        """O número de resultados deve respeitar top_k."""
        rec = PopularityRecommender()
        rec.fit(sample_interactions)

        assert len(rec.recommend(user_id="u1", top_k=2)) == 2
        assert len(rec.recommend(user_id="u1", top_k=1)) == 1

    def test_recommend_top_k_larger_than_catalog(
        self, sample_interactions: pd.DataFrame
    ) -> None:
        """top_k maior que o catálogo retorna todos os itens disponíveis."""
        rec = PopularityRecommender()
        rec.fit(sample_interactions)
        result = rec.recommend(user_id="u1", top_k=100)

        unique_items = sample_interactions["item_id"].nunique()
        assert len(result) == unique_items

    def test_recommend_ignores_user_id(self, sample_interactions: pd.DataFrame) -> None:
        """Popularidade não deve variar por usuário."""
        rec = PopularityRecommender()
        rec.fit(sample_interactions)

        assert rec.recommend("u1", top_k=3) == rec.recommend("u999", top_k=3)

    def test_is_instance_of_base_recommender(self) -> None:
        """PopularityRecommender deve satisfazer o contrato da interface."""
        rec = PopularityRecommender()
        assert isinstance(rec, BaseRecommender)
