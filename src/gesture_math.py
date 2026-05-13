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
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20


def _lm_dist(a: Any, b: Any) -> float:
    az = getattr(a, "z", None) or 0.0
    bz = getattr(b, "z", None) or 0.0
    return float(math.hypot(a.x - b.x, a.y - b.y, az - bz))


def hand_scale(landmarks: list[Any]) -> float:
    return max(_lm_dist(landmarks[WRIST], landmarks[MIDDLE_MCP]), _lm_dist(landmarks[WRIST], landmarks[INDEX_MCP]), 1e-5)


def pinch_ratio(landmarks: list[Any]) -> float:
    return _lm_dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_scale(landmarks)


def pinch_active(ratio: float, was_active: bool, closed_thr: float, open_thr: float) -> bool:
    """Histerese: fecha abaixo de closed_thr; só abre acima de open_thr (> closed_thr)."""
    if was_active:
        return ratio < open_thr
    return ratio < closed_thr


def is_peace_for_right_click(landmarks: list[Any], pinch_ratio_val: float, pinch_open_ratio: float) -> bool:
    """Indicador e médio estendidos; anelar e mindinho recolhidos; sem pinça (polegar longe do indicador)."""
    scale = hand_scale(landmarks)
    def span_tip_mcp(tip_i: int, mcp_i: int) -> float:
        return _lm_dist(landmarks[tip_i], landmarks[mcp_i]) / scale

    if span_tip_mcp(INDEX_TIP, INDEX_MCP) < 0.82 or span_tip_mcp(MIDDLE_TIP, MIDDLE_MCP) < 0.82:
        return False
    if span_tip_mcp(RING_TIP, RING_MCP) > 0.52 or span_tip_mcp(PINKY_TIP, PINKY_MCP) > 0.52:
        return False
    if pinch_ratio_val < pinch_open_ratio:
        return False
    return True
