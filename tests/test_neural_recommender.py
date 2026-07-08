"""Unit tests for the NeuralRecommender class.

Covers initialization, fitting, dynamic negative sampling, recommendations,
and checkpoint serialization.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from recsys.recommenders.base import BaseRecommender
from recsys.recommenders.neural import NeuralRecommender


@pytest.fixture()
def sample_interactions() -> pd.DataFrame:
    """Fixture providing a minimal dummy dataset of user-item interactions."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "item_id": ["i1", "i2", "i2", "i3", "i1", "i3", "i2", "i4"],
            "rating": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_neural_recommender_is_base_recommender() -> None:
    """NeuralRecommender must implement the BaseRecommender interface."""
    rec = NeuralRecommender()
    assert isinstance(rec, BaseRecommender)


def test_recommend_raises_before_fit() -> None:
    """recommend() must raise RuntimeError if fit() hasn't been called."""
    rec = NeuralRecommender()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        rec.recommend(user_id="u1")


def test_negative_sampling(sample_interactions: pd.DataFrame) -> None:
    """_sample_negatives must sample the requested ratio of negative items correctly."""
    rec = NeuralRecommender()

    # Setup mapping with a larger catalog of 50 items to ensure enough candidates
    catalog_items = [f"i{i}" for i in range(1, 51)]
    rec._item_idx = {it: i for i, it in enumerate(catalog_items)}
    rec._idx_to_item = {i: str(it) for it, i in rec._item_idx.items()}
    rec._user_idx = {
        u: i for i, u in enumerate(sample_interactions["user_id"].unique())
    }
    rec._idx_to_user = {i: u for u, i in rec._user_idx.items()}

    num_negatives = 3
    sampled_df = rec._sample_negatives(sample_interactions, num_negatives=num_negatives)

    # Check that ratings are present and split into 1.0 (positive) and 0.0 (negative)
    assert "rating" in sampled_df.columns
    positives = sampled_df[sampled_df["rating"] == 1.0]
    negatives = sampled_df[sampled_df["rating"] == 0.0]

    assert len(positives) == len(sample_interactions)
    assert len(negatives) == len(sample_interactions) * num_negatives

    # Verify no overlap: negative items for a user must not be in their positive set
    user_pos = sample_interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    for _, row in negatives.iterrows():
        u = row["user_id"]
        i = row["item_id"]
        assert i not in user_pos[u], (
            f"Negative item {i} found in user {u}'s positive interactions"
        )


def test_fit_and_recommend(sample_interactions: pd.DataFrame) -> None:
    """NeuralRecommender fit and recommend functionality test with a minimal model."""
    rec = NeuralRecommender(
        embedding_dim=8,
        mlp_hidden_dims=[16, 8],
        epochs=2,
        batch_size=4,
        lr=0.01,
        patience=1,
        val_split=0.2,
        seed=42,
    )

    rec.fit(sample_interactions)
    assert rec._is_fitted
    assert rec._model is not None

    # Recommend top 2
    recommendations = rec.recommend(user_id="u1", top_k=2)
    assert len(recommendations) == 2
    for item in recommendations:
        assert isinstance(item, str)
        assert item in rec._item_idx

    # Non-existent user returns empty list
    assert rec.recommend(user_id="non_existent_user", top_k=5) == []


def test_neural_recommender_respects_custom_num_negatives(
    sample_interactions: pd.DataFrame,
) -> None:
    """NeuralRecommender deve propagar num_negatives customizado para o
    sampler de negativos, não usar o valor hardcoded de 4."""
    recommender = NeuralRecommender(
        embedding_dim=4,
        epochs=1,
        num_negatives=7,
    )
    recommender.fit(sample_interactions)

    assert recommender.num_negatives == 7


def test_recommend_batch_uses_configurable_inference_batch_size(
    sample_interactions: pd.DataFrame,
) -> None:
    """recommend_batch deve aceitar um batch_size de inferência configurável
    em vez do valor hardcoded de 64, permitindo controlar o pico de memória."""
    recommender = NeuralRecommender(embedding_dim=4, epochs=1)
    recommender.fit(sample_interactions)

    user_ids = sample_interactions["user_id"].unique().tolist()
    results = recommender.recommend_batch(user_ids, top_k=5, inference_batch_size=2)

    assert set(results.keys()) == set(user_ids)
    assert all(len(v) <= 5 for v in results.values())


def test_save_and_load(sample_interactions: pd.DataFrame) -> None:
    """Verify that NeuralRecommender can be saved and loaded successfully."""
    rec = NeuralRecommender(
        embedding_dim=8,
        mlp_hidden_dims=[16, 8],
        epochs=2,
        batch_size=4,
        lr=0.01,
        patience=1,
        val_split=0.2,
        seed=42,
    )

    rec.fit(sample_interactions)
    original_recs = rec.recommend(user_id="u1", top_k=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pth"
        rec.save(checkpoint_path)

        # Load in a new recommender instance
        new_rec = NeuralRecommender()
        new_rec.load(checkpoint_path)

        assert new_rec._is_fitted
        assert new_rec.seed == rec.seed

        loaded_recs = new_rec.recommend(user_id="u1", top_k=2)
        assert loaded_recs == original_recs


def test_recommend_batch_matches_single(sample_interactions: pd.DataFrame) -> None:
    """Verify that recommend_batch yields identical recommendations to recommend()."""
    rec = NeuralRecommender(
        embedding_dim=8,
        mlp_hidden_dims=[16, 8],
        epochs=2,
        batch_size=4,
        lr=0.01,
        patience=1,
        val_split=0.2,
        seed=42,
    )
    rec.fit(sample_interactions)

    users = ["u1", "u2", "u3", "u4", "non_existent"]
    individual_recs = {u: rec.recommend(u, top_k=2) for u in users}
    batch_recs = rec.recommend_batch(users, top_k=2)

    assert batch_recs == individual_recs


def test_svd_recommend_batch_matches_single() -> None:
    """Verify that SVDRecommender.recommend_batch yields identical results to recommend()."""
    from recsys.recommenders.baseline import SVDRecommender

    # DataFrame com ratings variados para quebrar simetrias e evitar empates de score
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "item_id": ["i1", "i2", "i2", "i3", "i1", "i3", "i2", "i4"],
            "rating": [5.0, 1.2, 4.0, 3.1, 2.5, 4.8, 1.0, 5.0],
        }
    )
    rec = SVDRecommender(n_components=2, random_state=42)
    rec.fit(interactions)

    users = ["u1", "u2", "u3", "u4", "non_existent"]
    individual_recs = {u: rec.recommend(u, top_k=2) for u in users}
    batch_recs = rec.recommend_batch(users, top_k=2)

    assert batch_recs == individual_recs
