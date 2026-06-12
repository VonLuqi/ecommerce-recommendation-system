"""Métricas de avaliação para sistemas de recomendação.

Implementa funções para Precision@K, Recall@K, NDCG@K e MAP@K.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score


def precision_at_k(y_true: list[str], y_pred: list[str], k: int) -> float:
    """Calcula a Precision@K.

    Args:
        y_true: Lista de itens relevantes.
        y_pred: Lista de itens recomendados.
        k: Número de itens considerados.

    Returns:
        Valor da Precision@K.
    """
    if not y_true or not y_pred:
        return 0.0

    k = min(k, len(y_pred))
    y_pred_k = set(y_pred[:k])
    y_true_set = set(y_true)

    hits = len(y_pred_k.intersection(y_true_set))
    return float(hits / k)


def recall_at_k(y_true: list[str], y_pred: list[str], k: int) -> float:
    """Calcula o Recall@K.

    Args:
        y_true: Lista de itens relevantes.
        y_pred: Lista de itens recomendados.
        k: Número de itens considerados.

    Returns:
        Valor do Recall@K.
    """
    if not y_true:
        return 0.0

    k = min(k, len(y_pred))
    y_pred_k = set(y_pred[:k])
    y_true_set = set(y_true)

    hits = len(y_pred_k.intersection(y_true_set))
    return float(hits / len(y_true_set))


def ndcg_at_k(y_true: list[str], y_pred: list[str], k: int) -> float:
    """Calcula o Normalized Discounted Cumulative Gain (NDCG@K).

    Usa a implementação do scikit-learn.

    Args:
        y_true: Lista de itens relevantes.
        y_pred: Lista de itens recomendados ordenados por relevância.
        k: Número de itens considerados.

    Returns:
        Valor do NDCG@K.
    """
    if not y_true or not y_pred:
        return 0.0

    y_true_set = set(y_true)
    y_pred_k = y_pred[:k]

    all_items = list(y_true_set.union(set(y_pred_k)))

    true_relevance = np.array([1 if item in y_true_set else 0 for item in all_items])

    scores = np.zeros(len(all_items))
    for i, item in enumerate(y_pred_k):
        idx = all_items.index(item)
        scores[idx] = k - i

    return float(ndcg_score([true_relevance], [scores], k=k))


def map_at_k(y_true: list[str], y_pred: list[str], k: int) -> float:
    """Calcula o Mean Average Precision (MAP@K).

    Args:
        y_true: Lista de itens relevantes.
        y_pred: Lista de itens recomendados.
        k: Número de itens considerados.

    Returns:
        Valor do MAP@K.
    """
    if not y_true or not y_pred:
        return 0.0

    y_true_set = set(y_true)
    y_pred_k = y_pred[:k]

    hits = 0.0
    sum_precs = 0.0

    for i, item in enumerate(y_pred_k):
        if item in y_true_set:
            hits += 1.0
            sum_precs += hits / (i + 1.0)

    return float(sum_precs / min(len(y_true_set), k))
