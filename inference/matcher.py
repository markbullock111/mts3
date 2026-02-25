from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional
    faiss = None  # type: ignore


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return v
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))


@dataclass
class MatchCandidate:
    employee_id: int
    score: float
    embedding_id: int | None = None
    employee_code: str | None = None
    employee_name: str | None = None


class EmbeddingIndex:
    def search(self, query: np.ndarray, top_k: int = 5) -> list[MatchCandidate]:  # pragma: no cover - interface
        raise NotImplementedError


class NumpyEmbeddingIndex(EmbeddingIndex):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        if rows:
            self.matrix = np.stack([l2_normalize(np.asarray(r["embedding"], dtype=np.float32)) for r in rows], axis=0)
        else:
            self.matrix = np.empty((0, 0), dtype=np.float32)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[MatchCandidate]:
        if self.matrix.size == 0:
            return []
        q = l2_normalize(query).astype(np.float32)
        scores = self.matrix @ q
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k] if k > 0 else np.array([], dtype=np.int64)
        idx = idx[np.argsort(-scores[idx])]
        out: list[MatchCandidate] = []
        for i in idx.tolist():
            r = self.rows[i]
            out.append(
                MatchCandidate(
                    employee_id=int(r["employee_id"]),
                    score=float(scores[i]),
                    embedding_id=int(r.get("embedding_id")) if r.get("embedding_id") is not None else None,
                    employee_code=r.get("employee_code"),
                    employee_name=r.get("employee_name"),
                )
            )
        return out


class FaissEmbeddingIndex(EmbeddingIndex):
    def __init__(self, rows: list[dict]):
        if faiss is None:
            raise RuntimeError("faiss not available")
        self.rows = rows
        if not rows:
            self.index = None
            self.matrix_dim = 0
            return
        matrix = np.stack([l2_normalize(np.asarray(r["embedding"], dtype=np.float32)) for r in rows], axis=0)
        self.matrix_dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.matrix_dim)
        self.index.add(matrix)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[MatchCandidate]:
        if self.index is None:
            return []
        q = l2_normalize(query).astype(np.float32)[None, :]
        k = min(top_k, len(self.rows))
        scores, idxs = self.index.search(q, k)
        out: list[MatchCandidate] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            r = self.rows[idx]
            out.append(
                MatchCandidate(
                    employee_id=int(r["employee_id"]),
                    score=float(score),
                    embedding_id=int(r.get("embedding_id")) if r.get("embedding_id") is not None else None,
                    employee_code=r.get("employee_code"),
                    employee_name=r.get("employee_name"),
                )
            )
        return out


class EmbeddingMatcher:
    def __init__(self, rows: list[dict], prefer_faiss: bool = True):
        self.rows = rows
        self.prefer_faiss = prefer_faiss
        self.index: EmbeddingIndex
        if prefer_faiss and faiss is not None:
            try:
                self.index = FaissEmbeddingIndex(rows)
            except Exception:
                self.index = NumpyEmbeddingIndex(rows)
        else:
            self.index = NumpyEmbeddingIndex(rows)

    def best_match(self, embedding: np.ndarray, min_score: float, top_k: int = 5) -> MatchCandidate | None:
        cands = self.index.search(embedding, top_k=top_k)
        if not cands:
            return None
        cands = self._aggregate_by_employee(cands)
        best = cands[0] if cands else None
        if best and best.score >= min_score:
            return best
        return None

    def vote_match(self, embeddings: list[np.ndarray], min_score: float, top_k: int = 5) -> MatchCandidate | None:
        if not embeddings:
            return None
        scores_by_emp: dict[int, list[float]] = {}
        meta: dict[int, MatchCandidate] = {}
        for emb in embeddings:
            cands = self.index.search(emb, top_k=top_k)
            for c in cands:
                scores_by_emp.setdefault(c.employee_id, []).append(c.score)
                meta[c.employee_id] = c
        if not scores_by_emp:
            return None
        aggregated: list[MatchCandidate] = []
        for emp_id, scores in scores_by_emp.items():
            m = meta[emp_id]
            # temporal vote: average top scores plus mild support bonus by frequency
            top_scores = sorted(scores, reverse=True)[:3]
            agg = float(np.mean(top_scores) + 0.01 * min(len(scores), 5))
            aggregated.append(
                MatchCandidate(
                    employee_id=emp_id,
                    score=min(1.0, agg),
                    embedding_id=m.embedding_id,
                    employee_code=m.employee_code,
                    employee_name=m.employee_name,
                )
            )
        aggregated.sort(key=lambda x: x.score, reverse=True)
        best = aggregated[0]
        return best if best.score >= min_score else None

    @staticmethod
    def _aggregate_by_employee(cands: list[MatchCandidate]) -> list[MatchCandidate]:
        agg: dict[int, MatchCandidate] = {}
        for c in cands:
            existing = agg.get(c.employee_id)
            if existing is None or c.score > existing.score:
                agg[c.employee_id] = c
        out = list(agg.values())
        out.sort(key=lambda x: x.score, reverse=True)
        return out
