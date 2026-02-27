from __future__ import annotations

from pathlib import Path


_INSIGHTFACE_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "buffalo_l": (
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    )
}


def ensure_local_file(path: Path, model_label: str) -> None:
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"{model_label} not found at '{p}'. "
            "Offline mode forbids runtime downloads. Run 'python scripts/download_models.py' first."
        )


def validate_insightface_model_root(model_root: Path, model_name: str) -> Path:
    root = model_root.expanduser().resolve()
    model_name = model_name.strip()

    standard_model_dir = root / "models" / model_name
    direct_model_dir = root / model_name

    if standard_model_dir.is_dir():
        normalized_root = root
        model_dir = standard_model_dir
    elif direct_model_dir.is_dir():
        # Support callers that passed ".../insightface/models" instead of ".../insightface".
        normalized_root = root.parent
        model_dir = direct_model_dir
    else:
        raise FileNotFoundError(
            f"InsightFace model directory for '{model_name}' not found. Expected one of:\n"
            f"- {standard_model_dir}\n"
            f"- {direct_model_dir}\n"
            "Offline mode forbids runtime downloads. Run 'python scripts/download_models.py' first."
        )

    required = _INSIGHTFACE_REQUIRED_FILES.get(model_name.lower())
    if required:
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"InsightFace model '{model_name}' is incomplete at '{model_dir}'. Missing files: {', '.join(missing)}. "
                "Offline mode forbids runtime downloads. Re-download with 'python scripts/download_models.py'."
            )
    else:
        if not any(model_dir.glob("*.onnx")):
            raise FileNotFoundError(
                f"InsightFace model directory '{model_dir}' has no .onnx files. "
                "Offline mode forbids runtime downloads."
            )

    return normalized_root

