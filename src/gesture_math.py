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
THUMB_MCP = 2


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


def is_scroll_two_finger_pose(landmarks: list[Any]) -> bool:
    """Indicador e médio bem estendidos; anelar e mindinho bem recolhidos; polegar recolhido; sem pinças de clique."""
    scale = hand_scale(landmarks)
    if scale < 1e-4:
        return False

    def span_tip_mcp(tip_i: int, mcp_i: int) -> float:
        return _lm_dist(landmarks[tip_i], landmarks[mcp_i]) / scale

    if span_tip_mcp(INDEX_TIP, INDEX_MCP) < 0.84 or span_tip_mcp(MIDDLE_TIP, MIDDLE_MCP) < 0.84:
        return False
    if span_tip_mcp(RING_TIP, RING_MCP) > 0.47 or span_tip_mcp(PINKY_TIP, PINKY_MCP) > 0.47:
        return False
    if _lm_dist(landmarks[THUMB_TIP], landmarks[THUMB_MCP]) / scale > 0.48:
        return False
    if pinch_ratio(landmarks) < 0.48 or middle_thumb_pinch_ratio(landmarks) < 0.48:
        return False
    return True


def is_scroll_two_finger_pose_relaxed(landmarks: list[Any]) -> bool:
    """Igual à pose de scroll, com margens mais largas (mão ligeiramente rodada / fora do eixo Y). Só para uso com modo scroll já ativo."""
    scale = hand_scale(landmarks)
    if scale < 1e-4:
        return False

    def span_tip_mcp(tip_i: int, mcp_i: int) -> float:
        return _lm_dist(landmarks[tip_i], landmarks[mcp_i]) / scale

    if span_tip_mcp(INDEX_TIP, INDEX_MCP) < 0.62 or span_tip_mcp(MIDDLE_TIP, MIDDLE_MCP) < 0.62:
        return False
    if span_tip_mcp(RING_TIP, RING_MCP) > 0.68 or span_tip_mcp(PINKY_TIP, PINKY_MCP) > 0.68:
        return False
    if _lm_dist(landmarks[THUMB_TIP], landmarks[THUMB_MCP]) / scale > 0.70:
        return False
    if pinch_ratio(landmarks) < 0.28 or middle_thumb_pinch_ratio(landmarks) < 0.28:
        return False
    return True


def scroll_reference_y(landmarks: list[Any]) -> float:
    """Altura normalizada da mão para rolagem (pulso + base do médio, menos ruído que só o pulso)."""
    return 0.5 * (float(landmarks[WRIST].y) + float(landmarks[MIDDLE_MCP].y))


def scroll_two_finger_tips_y(landmarks: list[Any]) -> float:
    """Média da coordenada Y normalizada das pontas do indicador e do médio (rolar só mexendo estes dois dedos)."""
    return 0.5 * (float(landmarks[INDEX_TIP].y) + float(landmarks[MIDDLE_TIP].y))
