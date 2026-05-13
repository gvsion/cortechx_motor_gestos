"""Localização e download do modelo Hand Landmarker (.task) para a API Tasks."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

_DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def ensure_hand_landmarker_model(path: Path | None = None, url: str = _DEFAULT_URL) -> Path:
    """Garante que `path` exista; baixa do Google Storage se necessário."""
    path = path or default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except (OSError, urllib.error.URLError) as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Não foi possível baixar hand_landmarker.task (rede indisponível?). "
            f"Baixe manualmente de:\n  {url}\n  e salve em:\n  {path}"
        ) from e
    tmp.replace(path)
    return path
