from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .matcher import EmbeddingMatcher, MatchCandidate
from .model_checks import validate_insightface_model_root

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None  # type: ignore


@dataclass
class FaceCandidateFrame:
    crop_bgr: np.ndarray
    face_score: float
    sharpness: float
    ts_ms: int


@dataclass
class FaceTrackBuffer:
    max_items: int = 12
    items: list[FaceCandidateFrame] = field(default_factory=list)

    def add(self, crop_bgr: np.ndarray, face_score: float, ts_ms: int) -> None:
        sharpness = variance_of_laplacian(crop_bgr)
        self.items.append(FaceCandidateFrame(crop_bgr=crop_bgr.copy(), face_score=float(face_score), sharpness=float(sharpness), ts_ms=int(ts_ms)))
        self.items.sort(key=lambda x: (x.face_score, x.sharpness), reverse=True)
        if len(self.items) > self.max_items:
            self.items = self.items[: self.max_items]

    def top_k(self, k: int) -> list[FaceCandidateFrame]:
        ranked = sorted(self.items, key=lambda x: (x.face_score * 0.7 + min(x.sharpness / 200.0, 1.0) * 0.3), reverse=True)
        return ranked[:k]


def variance_of_laplacian(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceRecognizer:
    def __init__(self, model_root: str | Path, model_name: str = "buffalo_l", use_gpu: bool = True):
        if FaceAnalysis is None:
            raise RuntimeError("insightface is not installed")
        resolved_root = validate_insightface_model_root(Path(model_root), model_name=model_name)
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, root=str(resolved_root), providers=providers)
        try:
            self.app.prepare(ctx_id=0 if use_gpu else -1)
        except Exception:
            self.app.prepare(ctx_id=-1)

    def detect_faces(self, frame_bgr: np.ndarray):
        return self.app.get(frame_bgr)

    def add_best_face_from_person_crop(self, person_crop_bgr: np.ndarray, face_buffer: FaceTrackBuffer, ts_ms: int) -> bool:
        faces = self.detect_faces(person_crop_bgr)
        if not faces:
            return False
        best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0) or 0.0))
        bbox = getattr(best, "bbox", None)
        if bbox is None:
            return False
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2 = min(person_crop_bgr.shape[1], x2)
        y2 = min(person_crop_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return False
        crop = person_crop_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        face_buffer.add(crop, float(getattr(best, "det_score", 0.0) or 0.0), ts_ms)
        return True

    def embed_faces(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        embs: list[np.ndarray] = []
        for crop in crops:
            faces = self.detect_faces(crop)
            if not faces:
                continue
            best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0) or 0.0))
            emb = np.asarray(getattr(best, "embedding", []), dtype=np.float32)
            if emb.size == 0:
                continue
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embs.append(emb)
        return embs

    def identify_from_buffer(self, face_buffer: FaceTrackBuffer, matcher: EmbeddingMatcher, threshold: float, top_k_frames: int = 3) -> MatchCandidate | None:
        frames = face_buffer.top_k(top_k_frames)
        if not frames:
            return None
        embs = self.embed_faces([f.crop_bgr for f in frames])
        if not embs:
            return None
        return matcher.vote_match(embs, min_score=threshold, top_k=5)
