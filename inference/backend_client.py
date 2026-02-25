from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import requests


@dataclass
class BackendClient:
    base_url: str
    timeout: float = 5.0

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_settings(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/settings", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("values", {})

    def get_gallery(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/employees/gallery", timeout=max(self.timeout, 15))
        r.raise_for_status()
        return r.json()

    def post_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = requests.post(f"{self.base_url}/events", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def upload_enroll_frame(self, employee_id: int, kind: str, image_bgr: np.ndarray) -> dict[str, Any]:
        ok, enc = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image")
        files = [("files", ("capture.jpg", enc.tobytes(), "image/jpeg"))]
        url = f"{self.base_url}/employees/{employee_id}/enroll/{kind}"
        r = requests.post(url, files=files, timeout=max(self.timeout, 20))
        r.raise_for_status()
        return r.json()
