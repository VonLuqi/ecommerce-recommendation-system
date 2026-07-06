"""Testes leves para configuração e seeds.

Cobre:
- AppSettings carrega valores padrão corretamente.
- AppSettings lê variáveis de ambiente (override).
- AppSettings rejeita environment inválido.
- fix_seeds não lança exceção.
- fix_seeds é idempotente.
"""

from __future__ import annotations

import pytest

from recsys.config.settings import AppSettings
from recsys.utils.seeds import fix_seeds

# ---------------------------------------------------------------------------
# AppSettings
# ---------------------------------------------------------------------------


class TestAppSettings:
    """Testes unitários de AppSettings."""

    def test_default_project_name(self) -> None:
        """Valor padrão de project_name deve estar correto."""
        s = AppSettings()
        assert s.project_name == "ecommerce-recommendation-system"

    def test_default_environment_is_development(self) -> None:
        """Ambiente padrão deve ser 'development'."""
        s = AppSettings()
        assert s.environment == "development"

    def test_default_top_k(self) -> None:
        """top_k padrão deve ser 10."""
        s = AppSettings()
        assert s.model.top_k == 10

    def test_default_random_seed(self) -> None:
        """random_seed padrão deve ser 42."""
        s = AppSettings()
        assert s.model.random_seed == 42

    def test_env_override_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TOP_K via variável de ambiente deve sobrescrever o padrão."""
        monkeypatch.setenv("TOP_K", "20")
        s = AppSettings()
        assert s.model.top_k == 20

    def test_env_override_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENVIRONMENT via variável de ambiente deve ser aceito se válido."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        s = AppSettings()
        assert s.environment == "production"

    def test_invalid_environment_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENVIRONMENT inválido deve lançar ValidationError."""
        from pydantic import ValidationError

        monkeypatch.setenv("ENVIRONMENT", "chaos")
        with pytest.raises(ValidationError, match="environment deve ser"):
            AppSettings()

    def test_sub_models_are_loaded(self) -> None:
        """Os sub-modelos data, model e mlflow devem ser instâncias corretas."""
        from recsys.config.settings import DataSettings, MLflowSettings, ModelSettings

        s = AppSettings()
        assert isinstance(s.data, DataSettings)
        assert isinstance(s.model, ModelSettings)
        assert isinstance(s.mlflow, MLflowSettings)


# ---------------------------------------------------------------------------
# ModelSettings — recommender_type enum constraint
# ---------------------------------------------------------------------------


def test_recommender_type_rejects_invalid_value() -> None:
    """ModelSettings deve rejeitar valores de recommender_type fora do conjunto
    suportado (baseline, neural, popularity)."""
    from pydantic import ValidationError

    from recsys.config.settings import ModelSettings

    with pytest.raises(ValidationError):
        ModelSettings(RECOMMENDER_TYPE="invalid_type")


def test_recommender_type_accepts_valid_values() -> None:
    """ModelSettings deve aceitar os três valores suportados."""
    from recsys.config.settings import ModelSettings

    for valid in ("baseline", "neural", "popularity"):
        s = ModelSettings(RECOMMENDER_TYPE=valid)
        assert s.recommender_type == valid


# ---------------------------------------------------------------------------
# fix_seeds
# ---------------------------------------------------------------------------


class TestFixSeeds:
    """Testes unitários de fix_seeds."""

    def test_fix_seeds_does_not_raise(self) -> None:
        """fix_seeds(42) deve executar sem exceção."""
        fix_seeds(42)

    def test_fix_seeds_is_idempotent(self) -> None:
        """Chamar fix_seeds duas vezes com o mesmo seed deve ser seguro."""
        fix_seeds(0)
        fix_seeds(0)

    def test_fix_seeds_affects_random(self) -> None:
        """random.random() deve retornar o mesmo valor após seeds iguais."""
        import random

        fix_seeds(7)
        val1 = random.random()

        fix_seeds(7)
        val2 = random.random()

        assert val1 == val2

    def test_fix_seeds_affects_numpy(self) -> None:
        """numpy deve retornar valores determinísticos após fix_seeds."""
        import numpy as np

        fix_seeds(7)
        arr1 = np.random.rand(5).tolist()

        fix_seeds(7)
        arr2 = np.random.rand(5).tolist()

        assert arr1 == arr2
