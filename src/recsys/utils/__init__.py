"""Utilitários do projeto.

Expõe helpers reutilizáveis que não pertencem a nenhum subpacote
específico de domínio (dados, modelos, avaliação).
"""

from recsys.utils.seeds import fix_seeds

__all__ = ["fix_seeds"]
