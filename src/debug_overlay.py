"""Painel lateral do modo debug: etapas Mapear / Transladar / Interagir e mini-mapa da tela."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from src.pipeline import GestureMotorConfig, MotorOutput

PANEL_W = 420

_COLOR_TITLE = (220, 220, 220)
_COLOR_DIM = (140, 140, 145)
_SECTION_BGR = ((60, 180, 60), (40, 140, 220), (200, 120, 240))
_SCROLL_HEADER = (0, 165, 230)


def draw_scroll_feedback(vis_bgr: np.ndarray, out: MotorOutput, *, scroll_enabled: bool) -> None:
    """Indicadores no video: barra de armamento, borda modo scroll, medidor de velocidade, pulso ao rolar."""
    if not scroll_enabled:
        return
    sd = out.scroll_debug
    if sd is None:
        return
    fh, fw = vis_bgr.shape[:2]
    need = max(1, sd.steady_frames_needed)

    if sd.pose_matched and not sd.scroll_mode_active:
        ratio = min(1.0, sd.pose_streak / need)
        bw = int((fw - 24) * ratio)
        cv2.rectangle(vis_bgr, (12, 6), (fw - 12, 24), (45, 45, 50), -1)
        cv2.rectangle(vis_bgr, (12, 6), (12 + max(0, bw), 24), (0, 200, 255), -1)
        cv2.rectangle(vis_bgr, (12, 6), (fw - 12, 24), (90, 90, 95), 1, cv2.LINE_AA)
        cv2.putText(
            vis_bgr,
            f"SCROLL armando {sd.pose_streak}/{need}",
            (18, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )

    if sd.scroll_mode_active:
        cv2.rectangle(vis_bgr, (0, 0), (6, fh - 1), (60, 220, 80), -1)
        cx = fw - 12
        y0 = int(fh * 0.32)
        y1 = int(fh * 0.68)
        cv2.line(vis_bgr, (cx, y0), (cx, y1), (55, 55, 60), 2, cv2.LINE_AA)
        vmax = 0.012
        vn = max(-1.0, min(1.0, sd.ema_velocity / vmax))
        mid = (y0 + y1) // 2
        span = max(4, (y1 - y0) // 2 - 2)
        hfill = int(span * abs(vn))
        col_up = (80, 255, 120)
        col_dn = (120, 140, 255)
        if hfill > 0:
            if vn >= 0:
                cv2.line(vis_bgr, (cx, mid), (cx, mid - hfill), col_up, 4, cv2.LINE_AA)
            else:
                cv2.line(vis_bgr, (cx, mid), (cx, mid + hfill), col_dn, 4, cv2.LINE_AA)
        cv2.circle(vis_bgr, (cx, mid), 3, (220, 220, 225), -1, cv2.LINE_AA)

        bar_w = min(200, fw - 40)
        by = fh - 36
        filled = int(bar_w * sd.accum_fraction)
        cv2.rectangle(vis_bgr, (12, by), (12 + bar_w, by + 10), (40, 40, 45), -1)
        if filled > 0:
            cv2.rectangle(vis_bgr, (12, by), (12 + filled, by + 10), (0, 190, 255), -1)
        cv2.putText(
            vis_bgr,
            "acum. prox. linha",
            (16 + bar_w, by + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 185, 190),
            1,
            cv2.LINE_AA,
        )

    if out.scroll_dy != 0:
        msg = f"SCROLL  {out.scroll_dy:+d}  linha(s)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.95, 2
        (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
        x0 = max(8, (fw - tw) // 2)
        y0 = fh - 12
        cv2.rectangle(vis_bgr, (x0 - 12, y0 - th - 12), (x0 + tw + 12, y0 + 8), (0, 220, 255), -1)
        cv2.rectangle(vis_bgr, (x0 - 12, y0 - th - 12), (x0 + tw + 12, y0 + 8), (0, 120, 200), 2, cv2.LINE_AA)
        cv2.putText(vis_bgr, msg, (x0, y0), font, scale, (25, 25, 30), thick, cv2.LINE_AA)
    elif sd.scroll_mode_active:
        cv2.putText(
            vis_bgr,
            "SCROLL ativo — deslize a mao na vertical",
            (12, fh - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 230, 200),
            1,
            cv2.LINE_AA,
        )


def draw_comfort_zone(frame_bgr: np.ndarray, margin_norm: float) -> None:
    """Retângulo da região da imagem que mapeia para a tela inteira (margem normalizada)."""
    h, w = frame_bgr.shape[:2]
    mx = int(margin_norm * w)
    my = int(margin_norm * h)
    x1, y1 = max(0, mx), max(0, my)
    x2, y2 = w - 1 - mx, h - 1 - my
    if x2 > x1 and y2 > y1:
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 255), 1, cv2.LINE_AA)


def _put_lines(
    img: np.ndarray,
    x: int,
    y: int,
    lines: Sequence[str],
    color: tuple[int, int, int],
    *,
    scale: float = 0.4,
    line_step: int = 17,
) -> int:
    for line in lines:
        if not line:
            continue
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_step
    return y


def _section_header(img: np.ndarray, y: int, title: str, bar_bgr: tuple[int, int, int]) -> int:
    cv2.rectangle(img, (0, y - 14), (PANEL_W, y + 4), bar_bgr, -1)
    cv2.putText(img, title, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return y + 22


def _draw_minimap(
    panel: np.ndarray,
    x: int,
    y: int,
    box_w: int,
    box_h: int,
    sw: int,
    sh: int,
    cur: tuple[int, int] | None,
) -> None:
    cv2.rectangle(panel, (x, y), (x + box_w, y + box_h), (80, 80, 85), -1)
    cv2.rectangle(panel, (x, y), (x + box_w, y + box_h), (120, 120, 130), 1)
    if cur is None or sw < 1 or sh < 1:
        return
    u = cur[0] / max(sw - 1, 1)
    v = cur[1] / max(sh - 1, 1)
    px = x + 2 + int(u * max(box_w - 5, 1))
    py = y + 2 + int(v * max(box_h - 5, 1))
    cv2.circle(panel, (px, py), 5, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(panel, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)


def _build_debug_panel(
    h: int,
    out: MotorOutput,
    cfg: GestureMotorConfig,
    sw: int,
    sh: int,
    recent_events: Sequence[str],
    *,
    mouse_on: bool,
    touch_lines: Sequence[str] | None,
) -> np.ndarray:
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (42, 42, 44)

    y = 22
    cv2.putText(panel, "Pipeline (totem)", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TITLE, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(
        panel,
        f"Tela logica {sw}x{sh}  |  margem {cfg.margin_norm:.2f}",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        _COLOR_DIM,
        1,
        cv2.LINE_AA,
    )
    y += 26

    y = _section_header(panel, y, "1 MAPEAR — deteccao", _SECTION_BGR[0])
    lines1: list[str]
    if out.hand_count == 0:
        lines1 = ["Nenhuma mao detectada."]
    else:
        lines1 = [f"Maos: {out.hand_count} (usa a primeira)"]
        if out.tip_norm is not None:
            tx, ty = out.tip_norm
            lines1.append(f"Ponta do indicador (norm): ({tx:.3f}, {ty:.3f})")
        lines1.append("MediaPipe Hand Landmarker")
    if out.scroll_active:
        lines1.append("Scroll 2 dedos: clique/arrastar em pausa")
    y = _put_lines(panel, 10, y, lines1, (230, 230, 230))
    y += 10

    if cfg.scroll_enabled and out.scroll_debug is not None:
        y = _section_header(panel, y, "SCROLL — dois dedos", _SCROLL_HEADER)
        sd = out.scroll_debug
        sc_lines = [
            f"Pose: {'ok' if sd.pose_matched else 'nao'}   modo scroll: {'SIM' if sd.scroll_mode_active else 'nao'}",
            f"Armamento: {sd.pose_streak}/{sd.steady_frames_needed} frames estaveis",
            f"dy frame: {out.scroll_dy:+d}   raw_y: {sd.raw_delta_y:+.5f}   ema_v: {sd.ema_velocity:+.5f}",
            f"y ref (norm): {sd.reference_y:.4f}" if sd.reference_y is not None else "y ref: —",
        ]
        y = _put_lines(panel, 10, y, sc_lines, (220, 235, 240), scale=0.38, line_step=16)
        bar_w = PANEL_W - 24
        bh = 12
        bx, by = 12, y
        filled = int(bar_w * sd.accum_fraction)
        cv2.rectangle(panel, (bx, by), (bx + bar_w, by + bh), (48, 48, 52), -1)
        if filled > 0:
            cv2.rectangle(panel, (bx, by), (bx + filled, by + bh), (0, 200, 255), -1)
        cv2.rectangle(panel, (bx, by), (bx + bar_w, by + bh), (100, 100, 108), 1)
        cv2.putText(
            panel,
            f"Fila acumulada (fracao ate prox. linha): {sd.accum_fraction:.2f}",
            (bx, by + bh + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            _COLOR_DIM,
            1,
            cv2.LINE_AA,
        )
        y = by + bh + 30
    elif not cfg.scroll_enabled:
        y = _section_header(panel, y, "SCROLL", (60, 60, 65))
        y = _put_lines(panel, 10, y, ["Desligado (--no-scroll)."], _COLOR_DIM, scale=0.38, line_step=16)
        y += 6

    y = _section_header(panel, y, "2 TRANSLADAR — imagem -> totem", _SECTION_BGR[1])
    lines2 = [
        f"EMA alpha: {cfg.ema_alpha:.2f}",
        f"Max passo px: {cfg.max_step_pixels if cfg.max_step_pixels else 'off'}",
    ]
    md = out.mapping_debug
    if md is None:
        lines2.append("Sem dados de cursor neste frame.")
    else:
        a, b = md.tip_norm_xy
        lines2.append(f"Entrada (norm imagem): ({a:.3f}, {b:.3f})")
        lx, ly = md.linear01_xy
        lines2.append(f"Apos margem [0,1]: ({lx:.3f}, {ly:.3f})")
        sx2, sy2 = md.smooth01_xy
        lines2.append(f"Suavizado [0,1]: ({sx2:.3f}, {sy2:.3f})")
        cx, cy = md.screen_xy
        lines2.append(f"Pixels na tela: ({cx}, {cy})")
    if out.scroll_active and out.cursor_screen is not None:
        lines2.append("(Com scroll: cursor fixo no ultimo ponto)")
    y = _put_lines(panel, 10, y, lines2, (230, 230, 230))
    y += 6

    mini_w, mini_h = PANEL_W - 20, 96
    _draw_minimap(panel, 10, y, mini_w, mini_h, sw, sh, out.cursor_screen)
    y += mini_h + 4
    cv2.putText(panel, "Mini-mapa da tela do totem", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _COLOR_DIM, 1, cv2.LINE_AA)
    y += 22

    y = _section_header(panel, y, "3 INTERAGIR — gestos / clique", _SECTION_BGR[2])
    lines3: list[str] = []
    gd = out.gesture_debug
    if gd is not None:
        lines3.append(f"Pinch indice ativo: {'sim' if gd.index_pinch_active else 'nao'}")
        lines3.append(f"Pinch medio ativo: {'sim' if gd.middle_pinch_active else 'nao'}")
        lines3.append(f"Arrastar: {'sim' if gd.dragging else 'nao'}")
        if gd.pinch_elapsed_s is not None:
            lines3.append(f"Tempo pinch indice: {gd.pinch_elapsed_s:.2f}s")
        if gd.mid_pinch_elapsed_s is not None:
            lines3.append(f"Tempo pinch medio: {gd.mid_pinch_elapsed_s:.2f}s")
        lines3.append(f"Cooldown esq: {gd.cooldown_left_s:.2f}s  dir: {gd.right_cooldown_left_s:.2f}s")
    lines3.append(f"pynput (mouse real): {'ON' if mouse_on else 'off'}")
    if touch_lines:
        lines3.append("---")
        lines3.extend(touch_lines)
    y = _put_lines(panel, 10, y, lines3, (230, 230, 230))
    y += 8

    cv2.putText(panel, "Eventos neste frame:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_TITLE, 1, cv2.LINE_AA)
    y += 18
    if out.gesture_events:
        for ev in out.gesture_events:
            cv2.putText(
                panel,
                f"  {ev.kind.name} @ ({ev.screen_x},{ev.screen_y})",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (100, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 16
            if y > h - 80:
                break
    else:
        y = _put_lines(panel, 10, y, ["  (nenhum)"], _COLOR_DIM, scale=0.38, line_step=16)

    y += 6
    cv2.putText(panel, "Historico recente:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_TITLE, 1, cv2.LINE_AA)
    y += 18
    for hist in list(recent_events)[-6:]:
        cv2.putText(panel, hist[:54], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 200, 255), 1, cv2.LINE_AA)
        y += 14
        if y > h - 36:
            break

    cv2.putText(panel, "ESC sai", (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, _COLOR_DIM, 1, cv2.LINE_AA)
    return panel


def compose_debug_frame(
    vis_bgr: np.ndarray,
    out: MotorOutput,
    motor_cfg: GestureMotorConfig,
    sw: int,
    sh: int,
    recent_events: Sequence[str],
    *,
    mouse_on: bool,
    flash_text: str,
    t: float,
    flash_until: float,
    touch_lines: Sequence[str] | None = None,
) -> np.ndarray:
    """Desenha zona util no video, faixa de destaque opcional e concatena o painel a direita."""
    draw_comfort_zone(vis_bgr, motor_cfg.margin_norm)
    fh, fw = vis_bgr.shape[:2]
    draw_scroll_feedback(vis_bgr, out, scroll_enabled=motor_cfg.scroll_enabled)
    if flash_text and t < flash_until:
        cv2.putText(
            vis_bgr,
            flash_text,
            (max(16, fw // 2 - 110), fh // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    panel = _build_debug_panel(fh, out, motor_cfg, sw, sh, recent_events, mouse_on=mouse_on, touch_lines=touch_lines)
    return np.hstack([vis_bgr, panel])
