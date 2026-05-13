"""Rolagem: a pose de dois dedos ARMA um modo; o modo segue ativo com histerese até sair por frames inválidos."""

from __future__ import annotations

import dataclasses
from typing import Any

from src.gesture_math import (
    is_scroll_two_finger_pose,
    is_scroll_two_finger_pose_relaxed,
    scroll_two_finger_tips_y,
)


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
    invalid_exit_streak: int
    exit_frames_needed: int


@dataclasses.dataclass
class ScrollControllerConfig:
    """Entrada: N frames com pose de scroll. Saída: M frames seguidos fora da pose (ou sem mão)."""

    pose_steady_frames: int = 5
    pose_release_frames: int = 10
    dead_zone: float = 0.0016
    sensitivity: float = 18.0
    ema_alpha: float = 0.26
    max_lines_per_frame: int = 2
    max_raw_step: float = 0.0040


class ScrollController:
    """Modo scroll com trava; com modo ativo usa pose relaxada para não cortar o scroll por micro-desvios."""

    def __init__(self, config: ScrollControllerConfig | None = None) -> None:
        self._cfg = config or ScrollControllerConfig()
        self._latched = False
        self._pose_streak = 0
        self._invalid_exit = 0
        self._pending_reanchor = False
        self._prev_y: float | None = None
        self._ema_v = 0.0
        self._accum = 0.0
        self._last_debug = self._idle_debug()

    def _idle_debug(self) -> ScrollFrameDebug:
        rel = self._cfg.pose_release_frames
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
            invalid_exit_streak=0,
            exit_frames_needed=rel,
        )

    @property
    def last_frame_debug(self) -> ScrollFrameDebug:
        return self._last_debug

    def reset(self) -> None:
        self._latched = False
        self._pose_streak = 0
        self._invalid_exit = 0
        self._pending_reanchor = False
        self._prev_y = None
        self._ema_v = 0.0
        self._accum = 0.0
        self._last_debug = self._idle_debug()

    def _release_latch(self) -> None:
        self._latched = False
        self._pose_streak = 0
        self._invalid_exit = 0
        self._pending_reanchor = False
        self._prev_y = None
        self._ema_v = 0.0
        self._accum = 0.0

    def _mk_debug(
        self,
        *,
        pose_ok: bool,
        streak: int,
        need: int,
        mode_on: bool,
        y_ref: float | None,
        raw_in: float,
        lines: int,
        acc_frac: float,
        inv_exit: int,
        rel_need: int,
    ) -> ScrollFrameDebug:
        return ScrollFrameDebug(
            pose_matched=pose_ok,
            pose_streak=streak,
            steady_frames_needed=need,
            scroll_mode_active=mode_on,
            reference_y=y_ref,
            raw_delta_y=raw_in,
            ema_velocity=self._ema_v,
            accum_fraction=acc_frac,
            lines_this_frame=lines,
            invalid_exit_streak=inv_exit,
            exit_frames_needed=rel_need,
        )

    def update(
        self,
        landmarks: list[Any] | None,
        invert: float = 1.0,
    ) -> tuple[int, bool]:
        """Devolve (linhas_de_scroll, modo_scroll_ativo). O modo não exige pose válida em todos os frames."""
        need = self._cfg.pose_steady_frames
        rel_need = self._cfg.pose_release_frames

        if landmarks is None:
            if not self._latched:
                self.reset()
                return 0, False
            self._invalid_exit += 1
            self._pending_reanchor = True
            if self._invalid_exit >= rel_need:
                self._release_latch()
                self._last_debug = self._idle_debug()
                return 0, False
            self._last_debug = self._mk_debug(
                pose_ok=False,
                streak=self._pose_streak,
                need=need,
                mode_on=True,
                y_ref=None,
                raw_in=0.0,
                lines=0,
                acc_frac=min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0,
                inv_exit=self._invalid_exit,
                rel_need=rel_need,
            )
            return 0, True

        valid = (
            is_scroll_two_finger_pose_relaxed(landmarks)
            if self._latched
            else is_scroll_two_finger_pose(landmarks)
        )
        y_tips = scroll_two_finger_tips_y(landmarks)

        if not self._latched:
            if not valid:
                self._pose_streak = 0
                self._last_debug = self._mk_debug(
                    pose_ok=False,
                    streak=0,
                    need=need,
                    mode_on=False,
                    y_ref=None,
                    raw_in=0.0,
                    lines=0,
                    acc_frac=0.0,
                    inv_exit=0,
                    rel_need=rel_need,
                )
                return 0, False
            self._pose_streak += 1
            if self._pose_streak < need:
                self._last_debug = self._mk_debug(
                    pose_ok=True,
                    streak=self._pose_streak,
                    need=need,
                    mode_on=False,
                    y_ref=y_tips,
                    raw_in=0.0,
                    lines=0,
                    acc_frac=min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0,
                    inv_exit=0,
                    rel_need=rel_need,
                )
                return 0, False
            self._latched = True
            self._invalid_exit = 0
            self._pending_reanchor = False
            self._prev_y = y_tips
            self._ema_v = 0.0
            self._accum = 0.0
            self._last_debug = self._mk_debug(
                pose_ok=True,
                streak=self._pose_streak,
                need=need,
                mode_on=True,
                y_ref=y_tips,
                raw_in=0.0,
                lines=0,
                acc_frac=0.0,
                inv_exit=0,
                rel_need=rel_need,
            )
            return 0, True

        if not valid:
            self._invalid_exit += 1
            self._pending_reanchor = True
            if self._invalid_exit >= rel_need:
                self._release_latch()
                self._last_debug = self._idle_debug()
                return 0, False
            self._last_debug = self._mk_debug(
                pose_ok=False,
                streak=self._pose_streak,
                need=need,
                mode_on=True,
                y_ref=None,
                raw_in=0.0,
                lines=0,
                acc_frac=min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0,
                inv_exit=self._invalid_exit,
                rel_need=rel_need,
            )
            return 0, True

        self._invalid_exit = 0
        y = y_tips
        if self._pending_reanchor or self._prev_y is None:
            self._prev_y = y
            self._pending_reanchor = False
            self._last_debug = self._mk_debug(
                pose_ok=True,
                streak=self._pose_streak,
                need=need,
                mode_on=True,
                y_ref=y,
                raw_in=0.0,
                lines=0,
                acc_frac=min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0,
                inv_exit=0,
                rel_need=rel_need,
            )
            return 0, True

        raw_in = (self._prev_y - y) * invert
        cap = self._cfg.max_raw_step
        if cap > 0:
            raw_in = max(-cap, min(cap, raw_in))
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
        mcap = self._cfg.max_lines_per_frame
        if lines > mcap:
            lines = mcap
        elif lines < -mcap:
            lines = -mcap
        acc_bar = min(1.0, abs(self._accum) % 1.0) if abs(self._accum) > 1e-9 else 0.0
        self._last_debug = self._mk_debug(
            pose_ok=True,
            streak=self._pose_streak,
            need=need,
            mode_on=True,
            y_ref=y,
            raw_in=raw_in,
            lines=lines,
            acc_frac=acc_bar,
            inv_exit=0,
            rel_need=rel_need,
        )
        return lines, True
