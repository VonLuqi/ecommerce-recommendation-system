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

import mlflow
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

    Mantém arrays NumPy em memória para evitar overhead de tensores na indexação individual.
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        """Inicializa o dataset.

        Args:
            users: Array 1-D de índices de utilizadores.
            items: Array 1-D de índices de itens.
            ratings: Array 1-D de ratings (float).
        """
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[int, int, float]:
        return int(self.users[idx]), int(self.items[idx]), float(self.ratings[idx])


class _NegativeSamplingCollate:
    """Collation function personalizada para gerar negativos dinamicamente por batch.

    Evita instanciar todas as amostras negativas em memória de uma única vez.
    """

    def __init__(self, user_pos: list[set[int]], num_items: int, num_negatives: int = 4) -> None:
        self.user_pos = user_pos
        self.num_items = num_items
        self.num_negatives = num_negatives

    def __call__(
        self, batch: list[tuple[int, int, float]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(batch)
        total_len = batch_size * (1 + self.num_negatives)

        users = np.empty(total_len, dtype=np.int64)
        items = np.empty(total_len, dtype=np.int64)
        ratings = np.empty(total_len, dtype=np.float32)

        # Preenche os positivos
        for idx, (u, i, _r) in enumerate(batch):
            users[idx] = u
            items[idx] = i
            ratings[idx] = 1.0

        # Preenche os negativos
        write_idx = batch_size
        for u, _, _ in batch:
            pos_set = self.user_pos[u]
            sampled = 0
            attempts = 0
            while sampled < self.num_negatives and attempts < self.num_negatives * 2:
                random_items = np.random.randint(
                    0, self.num_items, size=(self.num_negatives - sampled) * 2
                )
                for item in random_items:
                    if item not in pos_set:
                        users[write_idx] = u
                        items[write_idx] = item
                        ratings[write_idx] = 0.0
                        write_idx += 1
                        sampled += 1
                        if sampled == self.num_negatives:
                            break
                attempts += 1

            # Fallback seguro caso não haja itens suficientes ou exceda tentativas
            if sampled < self.num_negatives:
                for _ in range(self.num_negatives - sampled):
                    users[write_idx] = u
                    items[write_idx] = 0
                    ratings[write_idx] = 0.0
                    write_idx += 1

        if write_idx < total_len:
            users = users[:write_idx]
            items = items[:write_idx]
            ratings = ratings[:write_idx]

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(ratings, dtype=torch.float32),
        )


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
        val_split: float = 0.02,
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
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

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

    def _sample_negatives_dataset(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        num_negatives: int = 4,
    ) -> _InteractionDataset:
        """Amostra itens negativos de forma altamente otimizada em NumPy.

        Retorna diretamente um _InteractionDataset para evitar overhead do pandas.
        """
        num_users = len(self._user_idx)
        num_items = len(self._item_idx)

        # Cria lista de sets para itens positivos de cada usuário
        user_pos = [set() for _ in range(num_users)]
        for u, i in zip(user_ids, item_ids, strict=False):
            user_pos[u].add(i)

        # Conta positivos por usuário
        user_pos_counts = np.bincount(user_ids, minlength=num_users)

        # Calcula o tamanho do dataset final (positivos + negativos)
        num_pos = len(user_ids)
        num_neg = num_pos * num_negatives
        total_len = num_pos + num_neg

        # Aloca arrays NumPy contíguos
        out_users = np.empty(total_len, dtype=np.int64)
        out_items = np.empty(total_len, dtype=np.int64)
        out_ratings = np.empty(total_len, dtype=np.float32)

        # Preenche os positivos no início
        out_users[:num_pos] = user_ids
        out_items[:num_pos] = item_ids
        out_ratings[:num_pos] = 1.0

        # Preenche os negativos a partir do final dos positivos
        write_idx = num_pos
        for u in range(num_users):
            n_pos = user_pos_counts[u]
            if n_pos == 0:
                continue

            n_neg_to_sample = n_pos * num_negatives
            pos_set = user_pos[u]

            sampled = 0
            attempts = 0
            # Amostra em lote para minimizar chamadas do gerador aleatório
            while sampled < n_neg_to_sample and attempts < n_neg_to_sample * 2:
                chunk_size = n_neg_to_sample - sampled
                random_items = np.random.randint(0, num_items, size=chunk_size * 2)
                for item in random_items:
                    if item not in pos_set:
                        out_users[write_idx] = u
                        out_items[write_idx] = item
                        out_ratings[write_idx] = 0.0
                        write_idx += 1
                        sampled += 1
                        pos_set.add(item)  # Evita duplicados no mesmo batch de negativos
                        if sampled == n_neg_to_sample:
                            break
                attempts += 1

        # Trunca caso não tenha conseguido amostrar o número total devido ao limite de tentativas
        if write_idx < total_len:
            out_users = out_users[:write_idx]
            out_items = out_items[:write_idx]
            out_ratings = out_ratings[:write_idx]

        return _InteractionDataset(out_users, out_items, out_ratings)

    def _sample_negatives(
        self,
        df: pd.DataFrame,
        num_negatives: int = 4,
    ) -> pd.DataFrame:
        """Amostra uniformemente itens negativos para cada utilizador no DataFrame.

        Compatibilidade com testes legados.
        """
        user_ids = df["user_id"].map(self._user_idx).to_numpy(dtype=np.int32)
        item_ids = df["item_id"].map(self._item_idx).to_numpy(dtype=np.int32)

        dataset = self._sample_negatives_dataset(user_ids, item_ids, num_negatives)

        users = [self._idx_to_user[u.item()] for u in dataset.users]
        items = [self._idx_to_item[i.item()] for i in dataset.items]
        ratings = [r.item() for r in dataset.ratings]

        return pd.DataFrame({
            "user_id": users,
            "item_id": items,
            "rating": ratings
        })

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o NeuMF com early stopping e amostragem dinâmica de negativos.

        1. Fixa seeds para reprodutibilidade.
        2. Mapeia IDs para índices contínuos.
        3. Converte interações para inteiros NumPy uma única vez.
        4. Divide em treino/validação.
        5. Amostra negativos fixos para validação.
        6. Instancia modelo, optimizador e loss.
        7. Loop de épocas com amostragem dinâmica de negativos em lote (on-the-fly) e early stopping.
        8. Restaura melhor estado do modelo.

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

        # Mapeamento para inteiros NumPy para alta performance
        user_ids_mapped = interactions["user_id"].map(self._user_idx).to_numpy(dtype=np.int32)
        item_ids_mapped = interactions["item_id"].map(self._item_idx).to_numpy(dtype=np.int32)

        # --- Split treino / validação ---
        indices = np.arange(len(interactions))
        np.random.shuffle(indices)
        split = int(len(indices) * (1.0 - self.val_split))

        train_user_ids = user_ids_mapped[indices[:split]]
        train_item_ids = item_ids_mapped[indices[:split]]

        val_user_ids = user_ids_mapped[indices[split:]]
        val_item_ids = item_ids_mapped[indices[split:]]

        # Cria lista de sets com os positivos de cada usuário (para verificação rápida de negativos)
        user_pos = [set() for _ in range(num_users)]
        for u, i in zip(user_ids_mapped, item_ids_mapped, strict=False):
            user_pos[u].add(i)

        # --- Datasets e DataLoaders de Validação (negativos fixos pré-gerados) ---
        _log.info("Gerando negativos para conjunto de validação...")
        val_dataset = self._sample_negatives_dataset(val_user_ids, val_item_ids, num_negatives=4)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
        )

        # --- Dataset e DataLoader de Treino (negativos gerados on-the-fly por batch) ---
        train_dataset = _InteractionDataset(
            train_user_ids, train_item_ids, np.ones(len(train_user_ids), dtype=np.float32)
        )
        train_collate = _NegativeSamplingCollate(user_pos, num_items, num_negatives=4)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=train_collate,
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
            "Iniciando treino NeuMF: %d users, %d items, %d val (com negs), "
            "device=%s",
            num_users,
            num_items,
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

            try:
                # Loga apenas se houver uma run ativa
                if mlflow.active_run():
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
            except Exception as e:
                _log.warning("Falha ao logar no MLflow: %s", e)

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

    def recommend_batch(
        self,
        user_ids: list[Any],
        top_k: int = 10,
    ) -> dict[Any, list[str]]:
        """Retorna recomendações top-K em lote de forma otimizada usando tensores do PyTorch.

        Args:
            user_ids: Lista de identificadores de utilizadores.
            top_k: Número de recomendações a retornar. Padrão: 10.

        Returns:
            Dicionário mapeando cada user_id para sua lista de recomendações.

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Chame fit() antes de recommend_batch().")

        known_users = [u for u in user_ids if u in self._user_idx]
        results = {u: [] for u in user_ids}

        if not known_users:
            return results

        # Processa usuários em blocos para evitar estourar a memória (CPU ou GPU)
        batch_size = 64
        num_items = len(self._item_idx)

        for i in range(0, len(known_users), batch_size):
            batch_u = known_users[i : i + batch_size]
            u_indices = [self._user_idx[u] for u in batch_u]

            # Expande tensores diretamente no dispositivo de destino, evitando overhead de alocação no host
            # e transferência de grandes volumes de dados via barramento CPU-GPU
            user_tensor = torch.tensor(u_indices, dtype=torch.long, device=self.device).repeat_interleave(num_items)
            item_tensor = self._all_item_ids_tensor.repeat(len(batch_u))

            with torch.no_grad():
                scores = self._model(user_tensor, item_tensor)
                scores = scores.cpu().numpy().reshape(len(batch_u), num_items)

            # Usa argpartition para encontrar os top-K maiores na CPU usando NumPy
            partition_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]

            for idx_in_batch, u_id in enumerate(batch_u):
                user_top_indices = partition_idx[idx_in_batch]
                user_scores = scores[idx_in_batch, user_top_indices]
                sorted_inner_idx = np.argsort(user_scores)[::-1]
                final_item_indices = user_top_indices[sorted_inner_idx]
                results[u_id] = [self._idx_to_item[idx] for idx in final_item_indices]

        return results

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
            ratings = ratings.to(device=self.device, dtype=torch.float32)

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
