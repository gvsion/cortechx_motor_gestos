"""Rolagem vertical: modo dois dedos (índice+médio, polegar recolhido) + deslocamento vertical."""

from __future__ import annotations

import dataclasses
from typing import Any

from src.gesture_math import is_scroll_two_finger_pose, scroll_reference_y


@dataclasses.dataclass(frozen=True)
class ScrollFrameDebug:
    """Instantâneo do controlador de rolagem para depuração e HUD."""

    pose_matched: bool
    pose_streak: int
    steady_frames_needed: int
    scroll_mode_active: bool
    reference_y: float | None
    raw_delta_y: float
    ema_velocity: float
    accum_fraction: float
    lines_this_frame: int


@dataclasses.dataclass
class ScrollControllerConfig:
    pose_steady_frames: int = 5
    dead_zone: float = 0.0010
    sensitivity: float = 48.0
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
        self._last_debug = self._idle_debug()

    def _idle_debug(self) -> ScrollFrameDebug:
        return ScrollFrameDebug(
            pose_matched=False,
            pose_streak=0,
            steady_frames_needed=self._cfg.pose_steady_frames,
            scroll_mode_active=False,
            reference_y=None,
            raw_delta_y=0.0,
            ema_velocity=0.0,
            accum_fraction=0.0,
            lines_this_frame=0,
        )

    @property
    def last_frame_debug(self) -> ScrollFrameDebug:
        return self._last_debug

    def reset(self) -> None:
        self._pose_streak = 0
        self._armed = False
        self._prev_y = None
        self._ema_v = 0.0
        self._accum = 0.0
        self._last_debug = self._idle_debug()

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
            self._last_debug = self._idle_debug()
            return 0, False

        self._pose_streak += 1
        y_ref = scroll_reference_y(landmarks)
        need = self._cfg.pose_steady_frames
        if self._pose_streak < need:
            self._last_debug = ScrollFrameDebug(
                pose_matched=True,
                pose_streak=self._pose_streak,
                steady_frames_needed=need,
                scroll_mode_active=False,
                reference_y=y_ref,
                raw_delta_y=0.0,
                ema_velocity=self._ema_v,
                accum_fraction=min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0,
                lines_this_frame=0,
            )
            return 0, False

        self._armed = True
        y = y_ref
        if self._prev_y is None:
            self._prev_y = y
            self._last_debug = ScrollFrameDebug(
                pose_matched=True,
                pose_streak=self._pose_streak,
                steady_frames_needed=need,
                scroll_mode_active=True,
                reference_y=y,
                raw_delta_y=0.0,
                ema_velocity=self._ema_v,
                accum_fraction=min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0,
                lines_this_frame=0,
            )
            return 0, True

        raw_in = (self._prev_y - y) * invert
        self._prev_y = y
        raw = raw_in
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
        acc_bar = min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0
        self._last_debug = ScrollFrameDebug(
            pose_matched=True,
            pose_streak=self._pose_streak,
            steady_frames_needed=need,
            scroll_mode_active=True,
            reference_y=y,
            raw_delta_y=raw_in,
            ema_velocity=self._ema_v,
            accum_fraction=acc_bar,
            lines_this_frame=lines,
        )
        return lines, True

    def _disarm(self) -> None:
        self._armed = False
        self._prev_y = None
        self._ema_v = 0.0
        self._accum = 0.0
        self._last_debug = self._idle_debug()
