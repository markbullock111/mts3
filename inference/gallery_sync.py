from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .backend_client import BackendClient
from .matcher import EmbeddingMatcher


@dataclass
class GalleryState:
    version: int = 0
    face_matcher: EmbeddingMatcher | None = None
    reid_matcher: EmbeddingMatcher | None = None


class GallerySyncService:
    def __init__(self, backend: BackendClient, refresh_seconds: int = 30):
        self.backend = backend
        self.refresh_seconds = max(5, refresh_seconds)
        self._state = GalleryState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sync_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            self.refresh_now()
        except Exception as exc:
            self._set_sync_error(exc)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_now()
                self._clear_sync_error()
            except Exception as exc:
                self._set_sync_error(exc)
            self._stop.wait(self.refresh_seconds)

    def refresh_now(self) -> None:
        payload = self.backend.get_gallery()
        version = int(payload.get("version", 0) or 0)
        with self._lock:
            if version == self._state.version and self._state.face_matcher is not None:
                return
        face_rows = payload.get("face_embeddings", [])
        reid_rows = payload.get("reid_embeddings", [])
        face_matcher = EmbeddingMatcher(face_rows, prefer_faiss=True)
        reid_matcher = EmbeddingMatcher(reid_rows, prefer_faiss=True)
        with self._lock:
            self._state = GalleryState(version=version, face_matcher=face_matcher, reid_matcher=reid_matcher)

    def _set_sync_error(self, exc: Exception) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        if msg == self._last_sync_error:
            return
        self._last_sync_error = msg
        print(
            f"[gallery] sync failed ({msg}). "
            f"Will retry every {self.refresh_seconds}s. "
            "Ensure backend is running."
        )

    def _clear_sync_error(self) -> None:
        if self._last_sync_error is not None:
            print("[gallery] sync restored.")
        self._last_sync_error = None

    def get_state(self) -> GalleryState:
        with self._lock:
            return self._state
