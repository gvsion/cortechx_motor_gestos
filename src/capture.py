"""Captura de vídeo da câmera com opção de espelhamento (modo espelho para o usuário)."""

from __future__ import annotations

import cv2


class CameraCapture:
    """Abre a câmera, lê frames BGR e aplica flip horizontal opcional."""

    def __init__(
        self,
        device_index: int = 0,
        mirror: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera (índice {device_index}).")
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._mirror = mirror

    @property
    def mirror(self) -> bool:
        return self._mirror

    @mirror.setter
    def mirror(self, value: bool) -> None:
        self._mirror = value

    def read_bgr(self):
        """Lê um frame BGR ou retorna (False, None) se falhar."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        if self._mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> CameraCapture:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
