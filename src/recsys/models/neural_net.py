"""Arquitetura NeuMF — Neural Matrix Factorization.

Combina Generalized Matrix Factorization (GMF) com uma Multi-Layer
Perceptron (MLP) num único modelo de recomendação neural.

A saída final é a concatenação dos ramos GMF e MLP passada por uma
camada linear final (NeuMF).

Referência:
    He et al., "Neural Collaborative Filtering", WWW 2017.
    https://arxiv.org/abs/1708.05031
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuMF(nn.Module):
    """Neural Matrix Factorization para recomendação.

    A arquitetura combina dois ramos:
        - **GMF** (Generalized Matrix Factorization): produto elemento-a-elemento
          entre os embeddings do usuário e do item.
        - **MLP** (Multi-Layer Perceptron): concatenação dos embeddings passada
          por camadas densas com activação ReLU e dropout.

    No final, os dois ramos são concatenados e projectados para um único
    logit de saída (relevância predita).

    Attributes:
        num_users: Número total de utilizadores no dataset.
        num_items: Número total de itens no dataset.
        embedding_dim: Dimensão dos embeddings de utilizador e item.
        mlp_hidden_dims: Lista com as dimensões das camadas ocultas da MLP.
        dropout: Taxa de dropout nas camadas MLP.
        use_gmf: Se ``True``, inclui o ramo GMF.
        use_mlp: Se ``True``, inclui o ramo MLP.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        use_gmf: bool = True,
        use_mlp: bool = True,
    ) -> None:
        """Inicializa o NeuMF.

        Args:
            num_users: Número total de utilizadores.
            num_items: Número total de itens.
            embedding_dim: Dimensão dos embeddings. Padrão: 64.
            mlp_hidden_dims: Dimensões das camadas ocultas MLP.
                Padrão: ``[256, 128, 64]``.
            dropout: Taxa de dropout. Padrão: 0.2.
            use_gmf: Activa ramo GMF. Padrão: ``True``.
            use_mlp: Activa ramo MLP. Padrão: ``True``.

        Raises:
            ValueError: Se ambos ``use_gmf`` e ``use_mlp`` forem ``False``.
        """
        super().__init__()

        if not use_gmf and not use_mlp:
            raise ValueError("Pelo menos um dos ramos (GMF ou MLP) deve estar activo.")

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp

        # --- Embeddings partilhados ---
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # --- Ramos ---
        self._build_gmf()
        self._build_mlp()

        # --- Camada final ---
        self._build_final_layer()

        # --- Inicialização de pesos ---
        self._init_weights()

    # ------------------------------------------------------------------
    # Construção dos ramos
    # ------------------------------------------------------------------

    def _build_gmf(self) -> None:
        """Constrói o ramo GMF.

        GMF: produto elemento-a-elemento entre user_embed e item_embed.
        """
        if self.use_gmf:
            # Camada linear do ramo GMF (dimensão = embedding_dim)
            self.gmf_layer = nn.Linear(self.embedding_dim, self.embedding_dim)

    def _build_mlp(self) -> None:
        """Constrói o ramo MLP.

        MLP: concatenação dos embeddings -> [dense + ReLU + Dropout] * N.
        """
        if not self.use_mlp:
            return

        mlp_modules: list[nn.Module] = []
        input_dim = self.embedding_dim * 2  # concatenação user + item

        for hidden_dim in self.mlp_hidden_dims:
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=self.dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*mlp_modules)

    def _build_final_layer(self) -> None:
        """Constrói a camada de saída final.

        Concatena os ramos GMF e MLP e projecta para 1 neurónio (logit).
        """
        gmf_out_dim = self.embedding_dim if self.use_gmf else 0
        mlp_out_dim = self.mlp_hidden_dims[-1] if self.use_mlp else 0
        final_in_dim = gmf_out_dim + mlp_out_dim

        self.final_layer = nn.Linear(final_in_dim, 1)

    # ------------------------------------------------------------------
    # Inicialização de pesos
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Inicializa pesos com distribuição normal N(0, 0.01).

        Utiliza a mesma estratégia do paper original NeuMF.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Forward pass: calcula logit de relevância para pares user-item.

        Args:
            user_ids: Tensor 1-D de índices de utilizadores (batch).
            item_ids: Tensor 1-D de índices de itens (batch).

        Returns:
            Tensor 1-D com logits de relevância (batch,).
        """
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        # Ramo GMF
        gmf_out: Tensor | None = None
        if self.use_gmf:
            gmf_vec = user_embed * item_embed  # elemento-a-elemento
            gmf_out = self.gmf_layer(gmf_vec)

        # Ramo MLP
        mlp_out: Tensor | None = None
        if self.use_mlp:
            mlp_vec = torch.cat([user_embed, item_embed], dim=-1)
            mlp_out = self.mlp_layers(mlp_vec)

        # Concatenação final
        if gmf_out is not None and mlp_out is not None:
            concat_vec = torch.cat([gmf_out, mlp_out], dim=-1)
        elif gmf_out is not None:
            concat_vec = gmf_out
        else:
            concat_vec = mlp_out  # type: ignore[assignment]

        logit = self.final_layer(concat_vec).squeeze(-1)
        return logit

    # ------------------------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------------------------

    def get_embedding_dims(self) -> dict[str, int]:
        """Devolve as dimensões dos embeddings do modelo.

        Returns:
            Dicionário com ``{"user": dim, "item": dim}``.
        """
        return {"user": self.embedding_dim, "item": self.embedding_dim}
