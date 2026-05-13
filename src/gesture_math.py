"""Distâncias e razões a partir dos 21 landmarks da mão (coordenadas normalizadas)."""

from __future__ import annotations

import math
from typing import Any


WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12


def _lm_dist(a: Any, b: Any) -> float:
    az = getattr(a, "z", None) or 0.0
    bz = getattr(b, "z", None) or 0.0
    return float(math.hypot(a.x - b.x, a.y - b.y, az - bz))


def hand_scale(landmarks: list[Any]) -> float:
    return max(_lm_dist(landmarks[WRIST], landmarks[MIDDLE_MCP]), _lm_dist(landmarks[WRIST], landmarks[INDEX_MCP]), 1e-5)


def pinch_ratio(landmarks: list[Any]) -> float:
    return _lm_dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_scale(landmarks)


def middle_thumb_pinch_ratio(landmarks: list[Any]) -> float:
    """Razão distância polegar–dedo médio / escala da mão (clique direito)."""
    return _lm_dist(landmarks[THUMB_TIP], landmarks[MIDDLE_TIP]) / hand_scale(landmarks)


def pinch_active(ratio: float, was_active: bool, closed_thr: float, open_thr: float) -> bool:
    """Histerese: fecha abaixo de closed_thr; só abre acima de open_thr (> closed_thr)."""
    if was_active:
        return ratio < open_thr
    return ratio < closed_thr
