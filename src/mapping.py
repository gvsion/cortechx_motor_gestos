"""Mapeia a ponta do indicador (normalizada na imagem) para coordenadas de tela com suavização."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

INDEX_FINGER_TIP = 8


def primary_index_tip_norm(hand_landmarks: list[list[Any]] | None) -> tuple[float, float] | None:
    """Retorna (x, y) normalizados [0, 1] da ponta do indicador da primeira mão detectada."""
    if not hand_landmarks:
        return None
    tip = hand_landmarks[0][INDEX_FINGER_TIP]
    return (float(tip.x), float(tip.y))


def probe_screen_size() -> tuple[int, int]:
    """Tenta obter resolução do monitor via Tk; fallback 1920x1080."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = max(1, int(root.winfo_screenwidth()))
        h = max(1, int(root.winfo_screenheight()))
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def _norm_to_01(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def tip_norm_to_linear01(tip_norm: tuple[float, float], margin_norm: float) -> tuple[float, float]:
    """Projeta a ponta normalizada na faixa [margem, 1-margem] para [0,1] (sem EMA)."""
    m = float(np.clip(margin_norm, 0.0, 0.49))
    x0, x1, y0, y1 = m, 1.0 - m, m, 1.0 - m
    nx, ny = tip_norm
    return _norm_to_01(nx, x0, x1), _norm_to_01(ny, y0, y1)


@dataclasses.dataclass(frozen=True)
class MapperFrameDebug:
    """Valores de um frame após margem, suavização e conversão para pixels da tela."""

    tip_norm_xy: tuple[float, float]
    linear01_xy: tuple[float, float]
    smooth01_xy: tuple[float, float]
    screen_xy: tuple[int, int]


@dataclasses.dataclass
class CursorMapperConfig:
    """A faixa [margem, 1-margem] na imagem (em X e Y) mapeia para a tela inteira.

    Margem maior = menos amplitude do dedo/braço para ir de canto a canto (útil no totem).
    """

    screen_width: int
    screen_height: int
    margin_norm: float = 0.26
    ema_alpha: float = 0.38
    max_step_pixels: float | None = 120.0


class HandToScreenMapper:
    """Converte ponta do indicador em coordenadas de tela; estado interno para EMA e passo máximo."""

    def __init__(self, config: CursorMapperConfig) -> None:
        self._cfg = config
        self._sw = max(1, int(config.screen_width))
        self._sh = max(1, int(config.screen_height))
        m = float(np.clip(config.margin_norm, 0.0, 0.49))
        self._x0, self._x1 = m, 1.0 - m
        self._y0, self._y1 = m, 1.0 - m
        self._alpha = float(np.clip(config.ema_alpha, 0.01, 1.0))
        self._ux: float | None = None
        self._uy: float | None = None
        self._prev_screen: tuple[int, int] | None = None
        self._last_debug: MapperFrameDebug | None = None

    @property
    def smoothed_norm01(self) -> tuple[float, float] | None:
        """Último (ux,uy) em [0,1] após EMA; None se o cursor ainda não foi inicializado."""
        if self._ux is None or self._uy is None:
            return None
        return (float(self._ux), float(self._uy))

    def reset(self) -> None:
        self._ux = self._uy = None
        self._prev_screen = None
        self._last_debug = None

    @property
    def last_frame_debug(self) -> MapperFrameDebug | None:
        return self._last_debug

    def update(self, tip_norm: tuple[float, float] | None) -> tuple[int, int] | None:
        """Recebe (x,y) na imagem normalizada ou None se não houver mão; devolve (sx,sy) na tela."""
        if tip_norm is None:
            self.reset()
            return None
        nx, ny = tip_norm
        tx = _norm_to_01(nx, self._x0, self._x1)
        ty = _norm_to_01(ny, self._y0, self._y1)
        if self._ux is None:
            self._ux, self._uy = tx, ty
        else:
            a = self._alpha
            self._ux = a * tx + (1.0 - a) * self._ux
            self._uy = a * ty + (1.0 - a) * self._uy
        assert self._ux is not None and self._uy is not None
        sx = int(round(self._ux * (self._sw - 1)))
        sy = int(round(self._uy * (self._sh - 1)))
        sx = int(np.clip(sx, 0, self._sw - 1))
        sy = int(np.clip(sy, 0, self._sh - 1))
        sx, sy = self._apply_max_step(sx, sy)
        self._prev_screen = (sx, sy)
        self._last_debug = MapperFrameDebug(
            tip_norm_xy=(float(nx), float(ny)),
            linear01_xy=(float(tx), float(ty)),
            smooth01_xy=(float(self._ux), float(self._uy)),
            screen_xy=(sx, sy),
        )
        return sx, sy

    def _apply_max_step(self, sx: int, sy: int) -> tuple[int, int]:
        cap = self._cfg.max_step_pixels
        if cap is None or self._prev_screen is None:
            return sx, sy
        px, py = self._prev_screen
        dx, dy = sx - px, sy - py
        dist = math.hypot(dx, dy)
        if dist <= cap or dist < 1e-6:
            return sx, sy
        s = cap / dist
        return int(px + dx * s), int(py + dy * s)

    def screen_to_preview_frame(self, sx: int, sy: int, frame_w: int, frame_h: int) -> tuple[int, int]:
        """Projeta ponto da tela no quadro do preview (mesma lógica que esticar a tela sobre o vídeo)."""
        fx = (sx / (self._sw - 1)) if self._sw > 1 else 0.0
        fy = (sy / (self._sh - 1)) if self._sh > 1 else 0.0
        return int(round(fx * (frame_w - 1))), int(round(fy * (frame_h - 1)))
