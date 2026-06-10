"""data — Carregadores de datasets de interação usuário-item."""

from recsys.data.instacart_loader import InstacartLoader
from recsys.data.loader import BaseInteractionLoader

__all__ = ["BaseInteractionLoader", "InstacartLoader"]
