"""Pipeline unificada do totem em tres etapas:

1. Mapear: HandTracker (MediaPipe) detecta maos e landmarks.
2. Transladar: HandToScreenMapper projeta a ponta do indicador da imagem para pixels da tela.
3. Interagir: GestureInteractor + ScrollController emitem cliques, arraste e rolagem.

O modulo expoe GestureMotor e MotorOutput para debug (run_debug) ou modo headless (run_totem).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from src.gesture_interactor import (
    GestureDebugState,
    GestureEvent,
    GestureInteractor,
    GestureInteractorConfig,
    GestureKind,
)
from src.hand_tracker import HandTracker, HandTrackerConfig
from src.mapping import CursorMapperConfig, HandToScreenMapper, MapperFrameDebug, primary_index_tip_norm, tip_norm_to_linear01
from src.scroll_control import ScrollController, ScrollControllerConfig


@dataclasses.dataclass
class GestureMotorConfig:
    """Parâmetros da tela e do mapeamento; scroll pode ser desligado no totem."""

    screen_width: int
    screen_height: int
    margin_norm: float = 0.15
    ema_alpha: float = 0.38
    max_step_pixels: float | None = 120.0
    scroll_enabled: bool = True
    scroll_invert: bool = False
    scroll_sensitivity: float | None = None


@dataclasses.dataclass
class MotorOutput:
    """Resultado de um frame processado (útil para UI de debug ou injeção de mouse)."""

    results: Any
    hand_count: int
    tip_norm: tuple[float, float] | None
    cursor_screen: tuple[int, int] | None
    gesture_events: list[GestureEvent]
    scroll_dy: int
    scroll_active: bool
    mapping_debug: MapperFrameDebug | None = None
    gesture_debug: GestureDebugState | None = None


class GestureMotor:
    """Encapsula HandTracker, mapeamento do cursor, gestos e scroll."""

    def __init__(
        self,
        motor_cfg: GestureMotorConfig,
        *,
        hand_cfg: HandTrackerConfig | None = None,
        gesture_cfg: GestureInteractorConfig | None = None,
    ) -> None:
        self._motor_cfg = motor_cfg
        max_step = motor_cfg.max_step_pixels
        if max_step is not None and max_step <= 0:
            max_step = None
        self._cursor = HandToScreenMapper(
            CursorMapperConfig(
                screen_width=motor_cfg.screen_width,
                screen_height=motor_cfg.screen_height,
                margin_norm=motor_cfg.margin_norm,
                ema_alpha=motor_cfg.ema_alpha,
                max_step_pixels=max_step,
            )
        )
        self._gestures = GestureInteractor(gesture_cfg or GestureInteractorConfig())
        scfg = ScrollControllerConfig()
        if motor_cfg.scroll_sensitivity is not None:
            scfg.sensitivity = motor_cfg.scroll_sensitivity
        self._scroll = ScrollController(scfg) if motor_cfg.scroll_enabled else None
        self._scroll_inv = -1.0 if motor_cfg.scroll_invert else 1.0
        self._tracker = HandTracker(hand_cfg or HandTrackerConfig())
        self._last_cur: tuple[int, int] | None = None
        self._prev_scroll_on = False

    def close(self) -> None:
        self._tracker.close()

    def __enter__(self) -> GestureMotor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def screen_to_preview(self, sx: int, sy: int, frame_w: int, frame_h: int) -> tuple[int, int]:
        return self._cursor.screen_to_preview_frame(sx, sy, frame_w, frame_h)

    def draw_landmarks(self, frame_bgr: Any, results: Any) -> Any:
        """Desenha landmarks no frame BGR (delega ao HandTracker)."""
        return self._tracker.draw_landmarks(frame_bgr, results)

    def process_rgb(self, image_rgb: Any, t: float) -> MotorOutput:
        """Executa detecção e lógica de interação num frame RGB uint8."""
        results = self._tracker.process(image_rgb)
        n = len(results.hand_landmarks) if results.hand_landmarks else 0
        first_lms = results.hand_landmarks[0] if n else None
        tip = primary_index_tip_norm(results.hand_landmarks)

        if self._scroll is not None and first_lms is not None:
            scroll_dy, scroll_on = self._scroll.update(first_lms, invert=self._scroll_inv)
        elif self._scroll is not None:
            self._scroll.reset()
            scroll_dy, scroll_on = 0, False
        else:
            scroll_dy, scroll_on = 0, False

        map_dbg: MapperFrameDebug | None
        if scroll_on:
            if not self._prev_scroll_on:
                events = self._gestures.reset()
            else:
                events = []
            self._prev_scroll_on = True
            cur = self._last_cur
            map_dbg = None
            if tip is not None and cur is not None:
                lin = tip_norm_to_linear01(tip, self._motor_cfg.margin_norm)
                sm = self._cursor.smoothed_norm01
                s01 = sm if sm is not None else lin
                map_dbg = MapperFrameDebug(
                    tip_norm_xy=(float(tip[0]), float(tip[1])),
                    linear01_xy=lin,
                    smooth01_xy=s01,
                    screen_xy=cur,
                )
        else:
            if self._prev_scroll_on:
                self._prev_scroll_on = False
            cur = self._cursor.update(tip)
            if cur is not None:
                self._last_cur = cur
            events = self._gestures.update(t, first_lms, cur)
            map_dbg = self._cursor.last_frame_debug

        gest_dbg = self._gestures.debug_state(t)

        return MotorOutput(
            results=results,
            hand_count=n,
            tip_norm=tip,
            cursor_screen=cur,
            gesture_events=events,
            scroll_dy=scroll_dy,
            scroll_active=scroll_on,
            mapping_debug=map_dbg,
            gesture_debug=gest_dbg,
        )


def apply_pynput_mouse(
    mouse: Any | None,
    out: MotorOutput,
) -> None:
    """Aplica movimento, cliques e rolagem ao sistema (pynput)."""
    if mouse is None:
        return
    cur = out.cursor_screen
    if cur is not None and not out.scroll_active:
        mouse.move(cur[0], cur[1])
    for ev in out.gesture_events:
        if ev.kind == GestureKind.LEFT_CLICK:
            mouse.left_click()
        elif ev.kind == GestureKind.RIGHT_CLICK:
            mouse.right_click()
        elif ev.kind == GestureKind.DRAG_START:
            mouse.left_down()
        elif ev.kind == GestureKind.DRAG_END:
            mouse.left_up()
    if out.scroll_dy != 0:
        mouse.scroll_vertical(out.scroll_dy)
