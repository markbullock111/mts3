from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


MODEL_URLS = {
    "yolo": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "insightface_buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def ensure_insightface(root: Path) -> None:
    zip_path = root / "models" / "insightface" / "buffalo_l.zip"
    target_dir = root / "models" / "insightface" / "models" / "buffalo_l"
    if target_dir.exists():
        print(f"[skip] {target_dir} already exists")
        return
    download(MODEL_URLS["insightface_buffalo_l"], zip_path)
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root / "models" / "insightface" / "models")


def main() -> None:
    p = argparse.ArgumentParser(description="Download model files for offline attendance runtime")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--skip-reid", action="store_true", help="Skip ReID model weights")
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    (root / "models" / "yolo").mkdir(parents=True, exist_ok=True)
    (root / "models" / "reid").mkdir(parents=True, exist_ok=True)
    (root / "models" / "insightface").mkdir(parents=True, exist_ok=True)

    download(MODEL_URLS["yolo"], root / "models" / "yolo" / "yolov8n.pt")
    ensure_insightface(root)
    if not args.skip_reid:
        download(MODEL_URLS["mobilenet_v3_small"], root / "models" / "reid" / "mobilenet_v3_small-047dcff4.pth")
    print("Done. Runtime can now operate offline (no downloads at runtime).")


if __name__ == "__main__":
    main()
