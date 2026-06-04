"""Utilitário para fixação determinística de seeds globais.

Garante reprodutibilidade dos experimentos fixando o estado aleatório de
todas as bibliotecas usadas no projeto em um único ponto.

A função deve ser chamada **antes** de qualquer operação aleatória:
criação de tensores, split de dados, inicialização de pesos, etc.

Usage:
    >>> from recsys.utils.seeds import fix_seeds
    >>> fix_seeds(42)

Referências:
    - https://pytorch.org/docs/stable/notes/randomness.html
    - https://numpy.org/doc/stable/reference/random/index.html
"""

from __future__ import annotations

import random


def fix_seeds(seed: int = 42) -> None:
    """Fixa seeds globais para reprodutibilidade.

    Cobre os módulos:
    - ``random`` (stdlib Python)
    - ``numpy``
    - ``torch`` (CPU e CUDA, se disponível)

    Args:
        seed: Valor inteiro para inicializar todos os geradores.
            Padrão: 42.

    Example:
        >>> fix_seeds(42)
        >>> import torch
        >>> torch.rand(1)  # resultado determinístico
        tensor([...])
    """
    random.seed(seed)

    # numpy — importado condicionalmente para evitar falha se não instalado
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    # torch — importado condicionalmente
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Garante determinismo nas operações convolucionais
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
