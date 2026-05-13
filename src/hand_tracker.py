"""Detecção de mãos e landmarks via MediaPipe Tasks (HandLandmarker)."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_module
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode

from src.hand_model import ensure_hand_landmarker_model

_BaseOptions = base_options_module.BaseOptions
_HandLandmarker = hand_landmarker_module.HandLandmarker
_HandLandmarkerOptions = hand_landmarker_module.HandLandmarkerOptions
_HandLandmarkerResult = hand_landmarker_module.HandLandmarkerResult
_HandLandmarksConnections = hand_landmarker_module.HandLandmarksConnections
_Image = image_module.Image
_ImageFormat = image_module.ImageFormat
_RunningMode = vision_task_running_mode.VisionTaskRunningMode


@dataclasses.dataclass(frozen=True)
class HandTrackerConfig:
    """Parâmetros do Hand Landmarker (Tasks)."""

    num_hands: int = 2
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: Path | None = None


class HandTracker:
    """HandLandmarker em modo vídeo: `process` recebe RGB uint8 e avança o relógio interno."""

    def __init__(self, config: HandTrackerConfig | None = None) -> None:
        self._config = config or HandTrackerConfig()
        model_path = self._config.model_path or ensure_hand_landmarker_model()
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.VIDEO,
            num_hands=self._config.num_hands,
            min_hand_detection_confidence=self._config.min_hand_detection_confidence,
            min_hand_presence_confidence=self._config.min_hand_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def process(self, image_rgb: np.ndarray) -> _HandLandmarkerResult:
        """Deteta mãos num frame RGB (H, W, 3) uint8; timestamps são monótonos em ms."""
        if image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb deve ser uint8.")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb deve ter forma (H, W, 3).")
        data = np.ascontiguousarray(image_rgb)
        mp_image = _Image(image_format=_ImageFormat.SRGB, data=data)
        if self._timestamp_ms == 0:
            self._timestamp_ms = int(time.monotonic() * 1000)
        else:
            self._timestamp_ms += 33
        return self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

    def draw_landmarks(self, frame_bgr: np.ndarray, results: _HandLandmarkerResult) -> np.ndarray:
        """Desenha esqueleto e pontos normalizados sobre uma cópia do frame BGR."""
        out = frame_bgr.copy()
        if not results.hand_landmarks:
            return out
        h, w = out.shape[:2]
        line_specs = _HandLandmarksConnections.HAND_CONNECTIONS
        for landmarks in results.hand_landmarks:
            for conn in line_specs:
                a = landmarks[conn.start]
                b = landmarks[conn.end]
                pt_a = (int(a.x * w), int(a.y * h))
                pt_b = (int(b.x * w), int(b.y * h))
                cv2.line(out, pt_a, pt_b, (255, 255, 255), 1, cv2.LINE_AA)
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
        return out


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
