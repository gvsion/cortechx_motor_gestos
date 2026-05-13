"""Rolagem vertical: modo dois dedos (índice+médio, polegar recolhido) + deslocamento vertical."""

from __future__ import annotations

import dataclasses
from typing import Any

from src.gesture_math import is_scroll_two_finger_pose, scroll_reference_y


@dataclasses.dataclass
class ScrollControllerConfig:
    pose_steady_frames: int = 6
    dead_zone: float = 0.0015
    sensitivity: float = 42.0
    ema_alpha: float = 0.42
    max_lines_per_frame: int = 8


class ScrollController:
    """Com o gesto estável por alguns frames, o movimento vertical da referência gera scroll."""

    def __init__(self, config: ScrollControllerConfig | None = None) -> None:
        self._cfg = config or ScrollControllerConfig()
        self._pose_streak = 0
        self._armed = False
        self._prev_y: float | None = None
        self._ema_v = 0.0
        self._accum = 0.0

    def reset(self) -> None:
        self._pose_streak = 0
        self._armed = False
        self._prev_y = None
        self._ema_v = 0.0
        self._accum = 0.0

    def update(
        self,
        landmarks: list[Any] | None,
        invert: float = 1.0,
    ) -> tuple[int, bool]:
        """Devolve (linhas_de_scroll, modo_scroll_ativo)."""
        if landmarks is None:
            self.reset()
            return 0, False

        if not is_scroll_two_finger_pose(landmarks):
            self._pose_streak = 0
            if self._armed:
                self._disarm()
            return 0, False

        self._pose_streak += 1
        if self._pose_streak < self._cfg.pose_steady_frames:
            return 0, False

        self._armed = True
        y = scroll_reference_y(landmarks)
        if self._prev_y is None:
            self._prev_y = y
            return 0, True

        raw = (self._prev_y - y) * invert
        self._prev_y = y
        if abs(raw) < self._cfg.dead_zone:
            raw = 0.0
        a = self._cfg.ema_alpha
        self._ema_v = a * raw + (1.0 - a) * self._ema_v
        self._accum += self._ema_v * self._cfg.sensitivity

        lines = int(self._accum)
        if lines != 0:
            self._accum -= float(lines)
        cap = self._cfg.max_lines_per_frame
        if lines > cap:
            lines = cap
        elif lines < -cap:
            lines = -cap
        return lines, True

    def _disarm(self) -> None:
        self._armed = False
        self._prev_y = None
        self._ema_v = 0.0
        self._accum = 0.0
