from __future__ import annotations

import numpy as np

from inference.matcher import EmbeddingMatcher, cosine_similarity


def test_cosine_similarity_basic():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == 1.0
    assert abs(cosine_similarity(a, c)) < 1e-6


def test_embedding_matcher_returns_best_employee():
    rows = [
        {"embedding_id": 1, "employee_id": 10, "employee_code": "E10", "employee_name": "A", "embedding": [1.0, 0.0, 0.0]},
        {"embedding_id": 2, "employee_id": 20, "employee_code": "E20", "employee_name": "B", "embedding": [0.0, 1.0, 0.0]},
        {"embedding_id": 3, "employee_id": 10, "employee_code": "E10", "employee_name": "A", "embedding": [0.9, 0.1, 0.0]},
    ]
    matcher = EmbeddingMatcher(rows, prefer_faiss=False)
    q = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    m = matcher.best_match(q, min_score=0.5)
    assert m is not None
    assert m.employee_id == 10
    assert m.score > 0.9


def test_temporal_vote_match_aggregates_frames():
    rows = [
        {"employee_id": 1, "embedding_id": 1, "embedding": [1.0, 0.0]},
        {"employee_id": 2, "embedding_id": 2, "embedding": [0.0, 1.0]},
    ]
    matcher = EmbeddingMatcher(rows, prefer_faiss=False)
    embs = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.95, 0.05], dtype=np.float32)]
    m = matcher.vote_match(embs, min_score=0.7)
    assert m is not None
    assert m.employee_id == 1
