"""Testes para o carregador InstacartLoader e funções do pipeline.

Usa arquivos CSV temporários para simular a estrutura real do Instacart
sem depender do dataset baixado.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from recsys.data.instacart_loader import InstacartLoader
from recsys.pipeline.feature_eng import encode_ids
from recsys.pipeline.preprocess import apply_kcore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def instacart_raw_dir(tmp_path: Path) -> Path:
    """Cria um diretório temporário com CSVs mínimos do Instacart."""
    orders_content = textwrap.dedent("""\
        order_id,user_id,eval_set,order_number
        1,u1,prior,1
        2,u1,prior,2
        3,u2,prior,1
        4,u3,prior,1
    """)
    order_products_content = textwrap.dedent("""\
        order_id,product_id,add_to_cart_order,reordered
        1,101,1,0
        1,102,2,0
        2,101,1,1
        3,101,1,0
        4,103,1,0
    """)

    (tmp_path / "orders.csv").write_text(orders_content)
    (tmp_path / "order_products__prior.csv").write_text(order_products_content)
    return tmp_path


@pytest.fixture()
def sample_interactions() -> pd.DataFrame:
    """DataFrame mínimo de interações para testes de pipeline."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i2", "i3", "i1", "i2", "i1"],
            "rating": [3.0, 2.0, 1.0, 2.0, 1.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# InstacartLoader
# ---------------------------------------------------------------------------


class TestInstacartLoader:
    """Testes unitários do carregador Instacart."""

    def test_load_returns_required_columns(self, instacart_raw_dir: Path) -> None:
        """load() deve retornar DataFrame com user_id, item_id e rating."""
        loader = InstacartLoader(instacart_raw_dir)
        df = loader.load()

        assert list(df.columns) == ["user_id", "item_id", "rating"]

    def test_load_aggregates_purchases(self, instacart_raw_dir: Path) -> None:
        """Frequência de compra deve ser a soma de ocorrências por par user-item."""
        loader = InstacartLoader(instacart_raw_dir)
        df = loader.load()

        # u1 comprou produto 101 nos pedidos 1 e 2 → rating = 2.0
        u1_101 = df[(df["user_id"] == "u1") & (df["item_id"] == "101")]
        assert not u1_101.empty
        assert u1_101["rating"].iloc[0] == 2.0

    def test_rating_dtype_is_float(self, instacart_raw_dir: Path) -> None:
        """A coluna rating deve ser do tipo float."""
        loader = InstacartLoader(instacart_raw_dir)
        df = loader.load()

        assert df["rating"].dtype == float

    def test_raises_if_orders_missing(self, tmp_path: Path) -> None:
        """FileNotFoundError se orders.csv não existir."""
        loader = InstacartLoader(tmp_path)
        with pytest.raises(FileNotFoundError, match="orders.csv"):
            loader.load()

    def test_raises_if_order_products_missing(self, tmp_path: Path) -> None:
        """FileNotFoundError se order_products__prior.csv não existir."""
        (tmp_path / "orders.csv").write_text("order_id,user_id\n1,u1\n")
        loader = InstacartLoader(tmp_path)
        with pytest.raises(FileNotFoundError, match="order_products__prior.csv"):
            loader.load()

    def test_unsupported_extension_on_path_object(self) -> None:
        """InstacartLoader aceita qualquer Path (valida os arquivos internamente)."""
        loader = InstacartLoader(Path("/nonexistent"))
        assert loader._raw_dir == Path("/nonexistent")

    def test_no_duplicates_in_result(self, instacart_raw_dir: Path) -> None:
        """Cada par (user_id, item_id) deve aparecer exatamente uma vez."""
        loader = InstacartLoader(instacart_raw_dir)
        df = loader.load()

        assert not df.duplicated(subset=["user_id", "item_id"]).any()


# ---------------------------------------------------------------------------
# apply_kcore (preprocess)
# ---------------------------------------------------------------------------


class TestApplyKcore:
    """Testes para a função de filtro k-core."""

    def test_removes_users_below_threshold(
        self, sample_interactions: pd.DataFrame
    ) -> None:
        """Usuários com menos de k interações devem ser removidos."""
        # u3 tem apenas 1 interação
        result = apply_kcore(sample_interactions, k=2)
        assert "u3" not in result["user_id"].values

    def test_removes_items_below_threshold(
        self, sample_interactions: pd.DataFrame
    ) -> None:
        """Itens com menos de k interações devem ser removidos."""
        # i3 tem apenas 1 interação (só u1)
        result = apply_kcore(sample_interactions, k=2)
        assert "i3" not in result["item_id"].values

    def test_kcore_1_returns_full_dataframe(
        self, sample_interactions: pd.DataFrame
    ) -> None:
        """k=1 não deve remover nenhuma linha."""
        result = apply_kcore(sample_interactions, k=1)
        assert len(result) == len(sample_interactions)


# ---------------------------------------------------------------------------
# encode_ids (feature_eng)
# ---------------------------------------------------------------------------


class TestEncodeIds:
    """Testes para a codificação de IDs."""

    def test_adds_user_idx_and_item_idx_columns(
        self, sample_interactions: pd.DataFrame
    ) -> None:
        """encode_ids deve adicionar as colunas user_idx e item_idx."""
        result = encode_ids(sample_interactions)
        assert "user_idx" in result.columns
        assert "item_idx" in result.columns

    def test_indices_are_contiguous(self, sample_interactions: pd.DataFrame) -> None:
        """Índices devem ser 0-based e contíguos."""
        result = encode_ids(sample_interactions)
        n_users = result["user_id"].nunique()
        n_items = result["item_id"].nunique()

        assert set(result["user_idx"].unique()) == set(range(n_users))
        assert set(result["item_idx"].unique()) == set(range(n_items))

    def test_original_columns_preserved(
        self, sample_interactions: pd.DataFrame
    ) -> None:
        """As colunas originais não devem ser alteradas."""
        result = encode_ids(sample_interactions)
        for col in ["user_id", "item_id", "rating"]:
            assert col in result.columns

    def test_does_not_mutate_input(self, sample_interactions: pd.DataFrame) -> None:
        """encode_ids não deve modificar o DataFrame original."""
        original_cols = list(sample_interactions.columns)
        encode_ids(sample_interactions)
        assert list(sample_interactions.columns) == original_cols
