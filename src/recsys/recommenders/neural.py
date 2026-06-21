"""Recomendador baseado em NeuMF (Neural Matrix Factorization) com PyTorch.

Estratégia concreta que implementa a interface ``BaseRecommender``
treinando uma rede NeuMF com early stopping e suporte a GPU (CUDA) se
disponível.

A classe lida com toda a orquestração:
    1. Mapeamento user/item IDs → índices contínuos.
    2. Criação de DataLoaders para treino e validação.
    3. Loop de treino com early stopping.
    4. Predição de scores para todos os itens não interagidos.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from recsys.models.neural_net import NeuMF
from recsys.recommenders.base import BaseRecommender
from recsys.utils.seeds import fix_seeds

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset PyTorch interno (privado ao módulo)
# ---------------------------------------------------------------------------


class _InteractionDataset(Dataset):
    """Dataset PyTorch para pares (user_id, item_id, rating).

    Constrói tensores directamente a partir de arrays NumPy para
    carregamento eficiente.
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        """Inicializa o dataset.

        Args:
            users: Array 1-D de índices de utilizadores.
            items: Array 1-D de índices de itens.
            ratings: Array 1-D de ratings (float).
        """
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------


class _EarlyStopping:
    """Interrompe o treino se a loss de validação não melhorar.

    Attributes:
        patience: Número de épocas sem melhoria antes de parar.
        min_delta: Variação mínima para considerar melhoria.
        best_loss: Melhor loss de validação observada.
        best_state: Estado do modelo na melhor época.
        counter: Épocas consecutivas sem melhoria.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        """Inicializa o Early Stopping.

        Args:
            patience: Épocas sem melhoria antes de parar. Padrão: 5.
            min_delta: Variação mínima para considerar melhoria. Padrão: 1e-4.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.best_state: dict[str, Any] | None = None
        self.counter: int = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Avalia a loss actual e actualiza contagem.

        Args:
            val_loss: Loss de validação da época actual.
            model: Modelo cujo estado deve ser salvo se melhorar.

        Returns:
            ``True`` se o treino deve parar, ``False`` caso contrário.
        """
        improvement = self.best_loss - val_loss

        if improvement > self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        _log.info(
            "EarlyStopping: %d/%d sem melhoria (best=%.6f, current=%.6f)",
            self.counter,
            self.patience,
            self.best_loss,
            val_loss,
        )
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Recomendador Neural
# ---------------------------------------------------------------------------


class NeuralRecommender(BaseRecommender):
    """Recomendador baseado em NeuMF com treino PyTorch e early stopping.

    Example:
        >>> import pandas as pd
        >>> rec = NeuralRecommender(
        ...     embedding_dim=32,
        ...     mlp_hidden_dims=[128, 64, 32],
        ...     epochs=50,
        ...     batch_size=256,
        ...     lr=0.001,
        ...     patience=5,
        ...     seed=42,
        ... )
        >>> interactions = pd.DataFrame({
        ...     "user_id": ["u1", "u1", "u2", "u2", "u2"],
        ...     "item_id": ["i1", "i2", "i1", "i3", "i4"],
        ...     "rating": [5.0, 3.0, 4.0, 2.0, 5.0],
        ... })
        >>> rec.fit(interactions)
        >>> rec.recommend(user_id="u1", top_k=2)
        ['i3', 'i4']
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        use_gmf: bool = True,
        use_mlp: bool = True,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        patience: int = 5,
        min_delta: float = 1e-4,
        val_split: float = 0.2,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        """Inicializa o recomendador neural.

        Args:
            embedding_dim: Dimensão dos embeddings. Padrão: 64.
            mlp_hidden_dims: Dimensões ocultas da MLP.
                Padrão: ``[256, 128, 64]``.
            dropout: Taxa de dropout. Padrão: 0.2.
            use_gmf: Activa ramo GMF. Padrão: ``True``.
            use_mlp: Activa ramo MLP. Padrão: ``True``.
            epochs: Número máximo de épocas. Padrão: 100.
            batch_size: Tamanho do batch. Padrão: 256.
            lr: Learning rate do Adam. Padrão: 0.001.
            weight_decay: Regularização L2. Padrão: 0.0.
            patience: Paciência do early stopping. Padrão: 5.
            min_delta: Variação mínima para melhoria. Padrão: 1e-4.
            val_split: Proporção dos dados para validação. Padrão: 0.2.
            seed: Semente global para reprodutibilidade. Padrão: 42.
            device: Dispositivo ('cuda', 'cpu' ou ``None`` para auto-detecção).
        """
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128, 64]

        self.embedding_dim = embedding_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.val_split = val_split
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Preenchido durante fit()
        self._model: NeuMF | None = None
        self._user_idx: dict[Any, int] = {}
        self._idx_to_user: dict[int, Any] = {}
        self._item_idx: dict[Any, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._all_item_ids_tensor: torch.Tensor | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit — Treino com Early Stopping
    # ------------------------------------------------------------------

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o NeuMF com early stopping.

        1. Fixa seeds para reprodutibilidade.
        2. Mapeia IDs para índices contínuos.
        3. Divide em treino/validação.
        4. Instancia modelo, optimizador e loss.
        5. Loop de épocas com early stopping.
        6. Restaura melhor estado do modelo.

        Args:
            interactions: DataFrame com colunas ``user_id``, ``item_id`` e ``rating``.
        """
        fix_seeds(self.seed)
        torch.manual_seed(self.seed)

        # --- Mapeamento de IDs ---
        users = interactions["user_id"].unique()
        items = interactions["item_id"].unique()

        self._user_idx = {u: i for i, u in enumerate(users)}
        self._idx_to_user = {i: u for u, i in self._user_idx.items()}
        self._item_idx = {it: i for i, it in enumerate(items)}
        self._idx_to_item = {i: str(it) for it, i in self._item_idx.items()}

        num_users = len(users)
        num_items = len(items)

        # --- Split treino / validação ---
        indices = np.arange(len(interactions))
        np.random.shuffle(indices)
        split = int(len(indices) * (1.0 - self.val_split))
        train_idx = indices[:split]
        val_idx = indices[split:]

        train_df = interactions.iloc[train_idx]
        val_df = interactions.iloc[val_idx]

        # --- Datasets e DataLoaders ---
        train_dataset = self._build_dataset(train_df)
        val_dataset = self._build_dataset(val_df)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
        )

        # --- Modelo ---
        self._model = NeuMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            mlp_hidden_dims=self.mlp_hidden_dims,
            dropout=self.dropout,
            use_gmf=self.use_gmf,
            use_mlp=self.use_mlp,
        ).to(self.device)

        optimizer = optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        # --- Early Stopping ---
        early_stopping = _EarlyStopping(
            patience=self.patience, min_delta=self.min_delta,
        )

        # --- Tensor de todos os itens (para recommend) ---
        self._all_item_ids_tensor = torch.arange(
            num_items, dtype=torch.long, device=self.device,
        )

        # --- Loop de treino ---
        _log.info(
            "Iniciando treino NeuMF: %d users, %d items, %d treino, %d val, "
            "device=%s",
            num_users,
            num_items,
            len(train_dataset),
            len(val_dataset),
            self.device,
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer, loss_fn, train=True)
            val_loss = self._run_epoch(val_loader, optimizer, loss_fn, train=False)

            _log.info(
                "Epoch %3d/%d — train_loss=%.6f — val_loss=%.6f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
            )

            # MLflow preparado: mlflow.log_metric("train_loss", train_loss, step=epoch)
            # MLflow preparado: mlflow.log_metric("val_loss", val_loss, step=epoch)

            if early_stopping.step(val_loss, self._model):
                _log.info(
                    "Early stopping activado na época %d. "
                    "Melhor val_loss: %.6f.",
                    epoch,
                    early_stopping.best_loss,
                )
                break

        # --- Restaurar melhor estado ---
        if early_stopping.best_state is not None:
            self._model.load_state_dict(early_stopping.best_state)
            _log.info(
                "Melhor estado do modelo restaurado (val_loss=%.6f).",
                early_stopping.best_loss,
            )

        self._model.eval()
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, user_id: Any, top_k: int = 10) -> list[str]:
        """Recomenda top-K itens para o utilizador.

        Calcula scores para todos os itens e retorna os ``top_k`` com
        maior relevância predita.

        Args:
            user_id: ID do utilizador alvo.
            top_k: Número de recomendações a retornar. Padrão: 10.

        Returns:
            Lista de item_ids recomendados, ordenados do mais relevante
            para o menos relevante.

        Raises:
            RuntimeError: Se o modelo não foi treinado (``fit()`` não chamado).
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Chame fit() antes de recommend().")

        if user_id not in self._user_idx:
            return []

        user_idx = self._user_idx[user_id]

        # --- Inferência em batch para todos os itens ---
        user_tensor = torch.full(
            (len(self._item_idx),),
            fill_value=user_idx,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            scores = self._model(user_tensor, self._all_item_ids_tensor)
            scores = scores.cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._idx_to_item[idx] for idx in top_indices]

    # ------------------------------------------------------------------
    # Salvar / Carregar
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Salva o modelo e metadados em disco.

        Args:
            path: Caminho do ficheiro ``.pth``.
        """
        if self._model is None:
            raise RuntimeError("Nada para salvar. Treine o modelo primeiro.")

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "model_config": {
                "num_users": self._model.num_users,
                "num_items": self._model.num_items,
                "embedding_dim": self._model.embedding_dim,
                "mlp_hidden_dims": self._model.mlp_hidden_dims,
                "dropout": self._model.dropout,
                "use_gmf": self._model.use_gmf,
                "use_mlp": self._model.use_mlp,
            },
            "mappings": {
                "user_idx": self._user_idx,
                "idx_to_user": self._idx_to_user,
                "item_idx": self._item_idx,
                "idx_to_item": self._idx_to_item,
            },
            "seed": self.seed,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        _log.info("Modelo salvo em '%s'.", path)

    def load(self, path: str | Path) -> None:
        """Carrega modelo e metadados do disco.

        Args:
            path: Caminho do ficheiro ``.pth``.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        cfg = checkpoint["model_config"]
        self._model = NeuMF(
            num_users=cfg["num_users"],
            num_items=cfg["num_items"],
            embedding_dim=cfg["embedding_dim"],
            mlp_hidden_dims=cfg["mlp_hidden_dims"],
            dropout=cfg["dropout"],
            use_gmf=cfg["use_gmf"],
            use_mlp=cfg["use_mlp"],
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._user_idx = checkpoint["mappings"]["user_idx"]
        self._idx_to_user = checkpoint["mappings"]["idx_to_user"]
        self._item_idx = checkpoint["mappings"]["item_idx"]
        self._idx_to_item = checkpoint["mappings"]["idx_to_item"]
        self.seed = checkpoint["seed"]

        self._all_item_ids_tensor = torch.arange(
            len(self._item_idx), dtype=torch.long, device=self.device,
        )
        self._is_fitted = True
        _log.info("Modelo carregado de '%s'.", path)

    # ------------------------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------------------------

    def _build_dataset(self, df: pd.DataFrame) -> _InteractionDataset:
        """Converte DataFrame em ``_InteractionDataset``.

        Args:
            df: DataFrame com user_id, item_id e rating.

        Returns:
            Dataset PyTorch.
        """
        users = df["user_id"].map(self._user_idx).to_numpy(dtype=np.int64)
        items = df["item_id"].map(self._item_idx).to_numpy(dtype=np.int64)
        ratings = df["rating"].to_numpy(dtype=np.float32)
        return _InteractionDataset(users, items, ratings)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        train: bool = True,
    ) -> float:
        """Executa uma época (treino ou validação).

        Args:
            loader: DataLoader com os dados.
            optimizer: Optimizador (ignorado se ``train=False``).
            loss_fn: Função de perda.
            train: Se ``True``, actualiza pesos. Padrão: ``True``.

        Returns:
            Loss média da época.
        """
        if train:
            self._model.train()
        else:
            self._model.eval()

        total_loss = 0.0
        num_batches = 0

        for user_ids, item_ids, ratings in loader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)

            if train:
                optimizer.zero_grad()

            outputs = self._model(user_ids, item_ids)
            loss = loss_fn(outputs, ratings)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)