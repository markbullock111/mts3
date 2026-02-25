from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .matcher import EmbeddingMatcher, MatchCandidate

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v3_small
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    T = None  # type: ignore
    mobilenet_v3_small = None  # type: ignore


@dataclass
class BodyCandidateFrame:
    crop_bgr: np.ndarray
    sharpness: float
    area: float
    ts_ms: int


@dataclass
class BodyTrackBuffer:
    max_items: int = 8
    items: list[BodyCandidateFrame] = field(default_factory=list)

    def add(self, crop_bgr: np.ndarray, ts_ms: int) -> None:
        if crop_bgr.size == 0:
            return
        h, w = crop_bgr.shape[:2]
        area = float(w * h)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.items.append(BodyCandidateFrame(crop_bgr=crop_bgr.copy(), sharpness=sharpness, area=area, ts_ms=ts_ms))
        self.items.sort(key=lambda x: (x.area, x.sharpness), reverse=True)
        if len(self.items) > self.max_items:
            self.items = self.items[: self.max_items]

    def top_k(self, k: int) -> list[BodyCandidateFrame]:
        return self.items[:k]


class ReIDRecognizer:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu"):
        if torch is None or mobilenet_v3_small is None or T is None:
            raise RuntimeError("torch/torchvision is not installed")
        self.device = torch.device(device)
        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier = torch.nn.Identity()
        if model_path and Path(model_path).exists():
            try:
                state = torch.load(str(model_path), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
                self.model.load_state_dict(state, strict=False)
            except Exception:
                pass
        self.model.eval().to(self.device)
        self.tx = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = self.tx(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            emb = y.flatten().detach().cpu().numpy().astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def identify_from_buffer(self, body_buffer: BodyTrackBuffer, matcher: EmbeddingMatcher, threshold: float, top_k_frames: int = 3) -> MatchCandidate | None:
        frames = body_buffer.top_k(top_k_frames)
        if not frames:
            return None
        embs = []
        for f in frames:
            try:
                embs.append(self.embed(f.crop_bgr))
            except Exception:
                continue
        if not embs:
            return None
        return matcher.vote_match(embs, min_score=threshold, top_k=5)
