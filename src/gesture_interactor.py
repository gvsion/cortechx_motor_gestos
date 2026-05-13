"""Máquina de gestos: pinça (clique / arrastar) e mão em V (clique direito)."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

from src import gesture_math as gm


class GestureKind(enum.Enum):
    LEFT_CLICK = enum.auto()
    RIGHT_CLICK = enum.auto()
    DRAG_START = enum.auto()
    DRAG_END = enum.auto()


@dataclasses.dataclass(frozen=True)
class GestureEvent:
    kind: GestureKind
    screen_x: int
    screen_y: int


@dataclasses.dataclass
class GestureInteractorConfig:
    pinch_closed: float = 0.34
    pinch_open: float = 0.48
    tap_min_s: float = 0.04
    tap_max_s: float = 0.39
    drag_hold_s: float = 0.42
    click_cooldown_s: float = 0.22
    peace_stable_frames: int = 10
    right_cooldown_s: float = 0.45


class GestureInteractor:
    """Consome landmarks da primeira mão + cursor em tela; emite eventos discretos."""

    def __init__(self, config: GestureInteractorConfig | None = None) -> None:
        self._cfg = config or GestureInteractorConfig()
        self._pinch_active = False
        self._pinch_t0: float | None = None
        self._dragging = False
        self._cooldown_until = 0.0
        self._r_cooldown_until = 0.0
        self._peace_frames = 0
        self._peace_latched = False

    def reset(self) -> list[GestureEvent]:
        out: list[GestureEvent] = []
        if self._dragging:
            out.append(GestureEvent(GestureKind.DRAG_END, 0, 0))
        self._pinch_active = False
        self._pinch_t0 = None
        self._dragging = False
        self._peace_frames = 0
        self._peace_latched = False
        return out

    def update(
        self,
        t: float,
        landmarks: list[Any] | None,
        screen_xy: tuple[int, int] | None,
    ) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        if landmarks is None or screen_xy is None:
            events.extend(self.reset())
            return events

        sx, sy = int(screen_xy[0]), int(screen_xy[1])
        old_pinch = self._pinch_active
        ratio = gm.pinch_ratio(landmarks)
        pinched = gm.pinch_active(ratio, old_pinch, self._cfg.pinch_closed, self._cfg.pinch_open)

        if pinched and not old_pinch:
            self._pinch_t0 = t

        if pinched and self._pinch_t0 is not None and not self._dragging:
            if (t - self._pinch_t0) >= self._cfg.drag_hold_s:
                self._dragging = True
                events.append(GestureEvent(GestureKind.DRAG_START, sx, sy))

        if not pinched and old_pinch:
            self._on_pinch_release(t, sx, sy, events)

        self._pinch_active = pinched

        if not pinched and not self._dragging and t >= self._r_cooldown_until and t >= self._cooldown_until:
            self._poll_right_click(landmarks, ratio, sx, sy, t, events)
        else:
            if not gm.is_peace_for_right_click(landmarks, ratio, self._cfg.pinch_open):
                self._peace_latched = False
                self._peace_frames = 0

        return events

    def _on_pinch_release(self, t: float, sx: int, sy: int, events: list[GestureEvent]) -> None:
        if self._pinch_t0 is None:
            return
        if self._dragging:
            self._dragging = False
            events.append(GestureEvent(GestureKind.DRAG_END, sx, sy))
        else:
            elapsed = t - self._pinch_t0
            if self._cfg.tap_min_s <= elapsed <= self._cfg.tap_max_s:
                events.append(GestureEvent(GestureKind.LEFT_CLICK, sx, sy))
                self._cooldown_until = t + self._cfg.click_cooldown_s
        self._pinch_t0 = None

    def _poll_right_click(
        self,
        landmarks: list[Any],
        ratio: float,
        sx: int,
        sy: int,
        t: float,
        events: list[GestureEvent],
    ) -> None:
        if gm.is_peace_for_right_click(landmarks, ratio, self._cfg.pinch_open):
            self._peace_frames += 1
        else:
            self._peace_frames = 0
            self._peace_latched = False
        if self._peace_latched:
            return
        if self._peace_frames >= self._cfg.peace_stable_frames:
            events.append(GestureEvent(GestureKind.RIGHT_CLICK, sx, sy))
            self._peace_latched = True
            self._peace_frames = 0
            self._r_cooldown_until = t + self._cfg.right_cooldown_s
