from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from inference.model_checks import validate_insightface_model_root

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - optional runtime dependency errors
    FaceAnalysis = None  # type: ignore

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v3_small
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    T = None  # type: ignore
    mobilenet_v3_small = None  # type: ignore


@dataclass
class FaceDetectionResult:
    embedding: np.ndarray
    det_score: float


class FaceEmbedder:
    def __init__(self, model_root: Path, model_name: str = "buffalo_l", use_gpu: bool = True):
        if FaceAnalysis is None:
            raise RuntimeError("insightface is not installed")
        resolved_root = validate_insightface_model_root(model_root, model_name=model_name)
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, root=str(resolved_root), providers=providers)
        ctx_id = 0 if use_gpu else -1
        self.app.prepare(ctx_id=ctx_id)

    def extract_from_bgr(self, image: np.ndarray) -> list[FaceDetectionResult]:
        faces = self.app.get(image)
        results: list[FaceDetectionResult] = []
        for face in faces:
            emb = np.asarray(face.embedding, dtype=np.float32)
            if emb.size == 0:
                continue
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            results.append(FaceDetectionResult(embedding=emb, det_score=score))
        return results


class ReIDEmbedder:
    def __init__(self, weights_path: Path | None = None, device: str = "cpu"):
        if torch is None or mobilenet_v3_small is None or T is None:
            raise RuntimeError("torch/torchvision is not installed")
        self.device = torch.device(device)
        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier = torch.nn.Identity()
        if weights_path and weights_path.exists():
            try:
                state = torch.load(str(weights_path), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    # best-effort local weights load
                    pass
            except Exception:
                pass
        self.model.eval().to(self.device)
        self.tx = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract_from_bgr(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = self.tx(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            emb = y.flatten().detach().cpu().numpy().astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
