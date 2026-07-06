"""Configuração central do projeto via Pydantic Settings.

Este módulo é a **fonte única de verdade** para todas as configurações do
projeto. Variáveis são carregadas na seguinte ordem de prioridade (maior
→ menor):

1. Variáveis de ambiente do sistema operacional.
2. Arquivo `.env` na raiz do projeto (se existir).
3. Valores padrão definidos no modelo.

Usage:
    >>> from recsys.config import settings
    >>> print(settings.project_name)
    ecommerce-recommendation-system

SOLID:
    - SRP: este módulo só gerencia configuração — sem lógica de negócio.
    - OCP: novos grupos de settings (model, data) são adicionados como
      sub-modelos sem alterar a interface existente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """Configurações relacionadas a caminhos de dados."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    raw_data_path: Path = Field(default=Path("data/raw"), alias="RAW_DATA_PATH")
    interim_data_path: Path = Field(
        default=Path("data/interim"), alias="INTERIM_DATA_PATH"
    )
    processed_data_path: Path = Field(
        default=Path("data/processed"), alias="PROCESSED_DATA_PATH"
    )
    dataset_name: str = Field(default="instacart", alias="DATASET_NAME")


class ModelSettings(BaseSettings):
    """Configurações do modelo e avaliação."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    top_k: int = Field(default=10, ge=1, alias="TOP_K")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    models_path: Path = Field(default=Path("models"), alias="MODELS_PATH")
    recommender_type: Literal["baseline", "neural", "popularity"] = Field(
        default="baseline", alias="RECOMMENDER_TYPE"
    )


class MLflowSettings(BaseSettings):
    """Configurações do MLflow tracking (ativo a partir da Etapa 8)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tracking_uri: str = Field(default="mlruns", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(
        default="neumf-instacart", alias="MLFLOW_EXPERIMENT_NAME"
    )


class AppSettings(BaseSettings):
    """Configurações globais da aplicação.

    Agrega todos os sub-modelos de configuração em um ponto único.
    Consumida diretamente por treino, avaliação, pipeline e scripts.

    Attributes:
        project_name: Nome do projeto.
        environment: Ambiente de execução (development | staging | production).
        data: Configurações de dados.
        model: Configurações do modelo.
        mlflow: Configurações do MLflow.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    project_name: str = Field(
        default="ecommerce-recommendation-system", alias="PROJECT_NAME"
    )
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Sub-configurações compostas (não mapeadas de env vars diretamente)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)

    @model_validator(mode="after")
    def _validate_environment(self) -> AppSettings:
        """Garante que o ambiente tem um valor válido.

        Returns:
            A instância validada.

        Raises:
            ValueError: Se ``environment`` não for um valor permitido.
        """
        allowed = {"development", "staging", "production"}
        if self.environment not in allowed:
            raise ValueError(
                f"environment deve ser um de {allowed}, recebido: '{self.environment}'"
            )
        return self
