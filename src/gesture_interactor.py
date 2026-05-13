"""Máquina de gestos: pinça índice (clique esquerdo / arrastar) e pinça médio (clique direito)."""

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


@dataclasses.dataclass(frozen=True)
class GestureDebugState:
    """Instantâneo da máquina de gestos para depuração na UI."""

    index_pinch_active: bool
    middle_pinch_active: bool
    dragging: bool
    pinch_elapsed_s: float | None
    mid_pinch_elapsed_s: float | None
    cooldown_left_s: float
    right_cooldown_left_s: float


@dataclasses.dataclass
class GestureInteractorConfig:
    pinch_closed: float = 0.34
    pinch_open: float = 0.48
    right_pinch_closed: float = 0.36
    right_pinch_open: float = 0.50
    tap_min_s: float = 0.04
    tap_max_s: float = 0.39
    drag_hold_s: float = 0.42
    click_cooldown_s: float = 0.22
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
        self._mid_pinch_active = False
        self._mid_pinch_t0: float | None = None

    def debug_state(self, t: float) -> GestureDebugState:
        pinch_elapsed = None if self._pinch_t0 is None else max(0.0, t - self._pinch_t0)
        mid_elapsed = None if self._mid_pinch_t0 is None else max(0.0, t - self._mid_pinch_t0)
        return GestureDebugState(
            index_pinch_active=self._pinch_active,
            middle_pinch_active=self._mid_pinch_active,
            dragging=self._dragging,
            pinch_elapsed_s=pinch_elapsed,
            mid_pinch_elapsed_s=mid_elapsed,
            cooldown_left_s=max(0.0, self._cooldown_until - t),
            right_cooldown_left_s=max(0.0, self._r_cooldown_until - t),
        )

    def reset(self) -> list[GestureEvent]:
        out: list[GestureEvent] = []
        if self._dragging:
            out.append(GestureEvent(GestureKind.DRAG_END, 0, 0))
        self._pinch_active = False
        self._pinch_t0 = None
        self._dragging = False
        self._mid_pinch_active = False
        self._mid_pinch_t0 = None
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
            self._reset_middle_pinch()
            self._pinch_t0 = t

        if pinched and self._pinch_t0 is not None and not self._dragging:
            if (t - self._pinch_t0) >= self._cfg.drag_hold_s:
                self._dragging = True
                events.append(GestureEvent(GestureKind.DRAG_START, sx, sy))

        if not pinched and old_pinch:
            self._on_pinch_release(t, sx, sy, events)

        self._pinch_active = pinched

        if (
            not pinched
            and not self._dragging
            and t >= self._r_cooldown_until
            and t >= self._cooldown_until
        ):
            self._update_middle_thumb_pinch(t, landmarks, sx, sy, pinched, events)

        return events

    def _reset_middle_pinch(self) -> None:
        self._mid_pinch_active = False
        self._mid_pinch_t0 = None

    def _update_middle_thumb_pinch(
        self,
        t: float,
        landmarks: list[Any],
        sx: int,
        sy: int,
        index_pinched: bool,
        events: list[GestureEvent],
    ) -> None:
        mid_ratio = gm.middle_thumb_pinch_ratio(landmarks)
        old_mid = self._mid_pinch_active
        mid_pinched = gm.pinch_active(
            mid_ratio,
            old_mid,
            self._cfg.right_pinch_closed,
            self._cfg.right_pinch_open,
        )
        if mid_pinched and not old_mid:
            self._mid_pinch_t0 = t
        if not mid_pinched and old_mid:
            self._on_middle_pinch_release(t, sx, sy, index_pinched, events)
        self._mid_pinch_active = mid_pinched

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

    def _on_middle_pinch_release(
        self,
        t: float,
        sx: int,
        sy: int,
        index_pinched: bool,
        events: list[GestureEvent],
    ) -> None:
        if self._mid_pinch_t0 is None:
            return
        if index_pinched:
            self._mid_pinch_t0 = None
            return
        elapsed = t - self._mid_pinch_t0
        if self._cfg.tap_min_s <= elapsed <= self._cfg.tap_max_s:
            events.append(GestureEvent(GestureKind.RIGHT_CLICK, sx, sy))
            self._r_cooldown_until = t + self._cfg.right_cooldown_s
        self._mid_pinch_t0 = None
