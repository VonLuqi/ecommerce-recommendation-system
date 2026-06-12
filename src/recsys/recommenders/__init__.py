"""recommenders — Estratégias de recomendação de produtos.

Exportações públicas do subpacote.
"""

from recsys.recommenders.base import BaseRecommender
from recsys.recommenders.baseline import SVDRecommender
from recsys.recommenders.popularity import PopularityRecommender

__all__ = ["BaseRecommender", "SVDRecommender", "PopularityRecommender"]
