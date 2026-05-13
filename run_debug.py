#!/usr/bin/env python3
"""Loop de debug: câmera + MediaPipe Hands + visualização dos landmarks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _venv_python() -> Path | None:
    for name in ("python3", "python"):
        p = ROOT / ".venv" / "bin" / name
        if p.is_file():
            return p
    return None


def _reexec_with_venv() -> None:
    vpy = _venv_python()
    if vpy is None:
        return
    if Path(sys.executable).resolve() == vpy.resolve():
        return
    script = Path(__file__).resolve()
    os.execv(str(vpy), [str(vpy), str(script), *sys.argv[1:]])


try:
    import cv2
except ModuleNotFoundError:
    _reexec_with_venv()
    try:
        import cv2
    except ModuleNotFoundError:
        vpy = _venv_python()
        in_venv = vpy is not None and Path(sys.executable).resolve() == vpy.resolve()
        lines = [
            "OpenCV (cv2) não está instalado neste interpretador.",
            f"  Interpretador: {sys.executable}",
            "",
        ]
        if in_venv:
            lines.extend(
                [
                    "O ambiente virtual está ativo, mas falta o pacote opencv-python.",
                    "Instale as dependências do projeto:",
                    f"    {vpy} -m pip install -r {ROOT / 'requirements.txt'}",
                    "",
                    "No Cursor/VS Code: confirme que o interpretador selecionado é o .venv do projeto.",
                ]
            )
        else:
            lines.extend(
                [
                    "Crie o ambiente e instale as dependências:",
                    f"    cd {ROOT}",
                    "    python3 -m venv .venv",
                    "    . .venv/bin/activate",
                    "    pip install -r requirements.txt",
                    "    python run_debug.py",
                    "",
                    "Ou use diretamente o Python do venv (se já existir):",
                    f"    {ROOT / '.venv' / 'bin' / 'python3'} run_debug.py",
                ]
            )
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(1) from None

from src.capture import CameraCapture
from src.debug_overlay import compose_debug_frame
from src.gesture_interactor import GestureKind
from src.gesture_math import is_scroll_two_finger_pose, middle_thumb_pinch_ratio, pinch_ratio
from src.hand_tracker import bgr_to_rgb
from src.mapping import probe_screen_size
from src.pipeline import GestureMotor, GestureMotorConfig, apply_pynput_mouse

import numpy as np


def _letterbox_to_display(img: np.ndarray, disp_w: int, disp_h: int) -> np.ndarray:
    """Escala para caber em disp_w x disp_h mantendo proporção (barras pretas se preciso)."""
    h, w = img.shape[:2]
    dw, dh = max(1, disp_w), max(1, disp_h)
    scale = min(dw / max(w, 1), dh / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if nw == dw and nh == dh:
        return resized
    canvas = np.zeros((dh, dw, 3), dtype=np.uint8)
    ox, oy = (dw - nw) // 2, (dh - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug: câmera e landmarks das mãos.")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão 0).")
    parser.add_argument("--no-mirror", action="store_true", help="Desativa flip horizontal.")
    parser.add_argument("--width", type=int, default=None, help="Largura solicitada ao driver.")
    parser.add_argument("--height", type=int, default=None, help="Altura solicitada ao driver.")
    parser.add_argument("--screen-width", type=int, default=None, help="Largura lógica da tela do totem (padrão: detectar).")
    parser.add_argument("--screen-height", type=int, default=None, help="Altura lógica da tela do totem (padrão: detectar).")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.12,
        help="Margem normalizada [0,0.49] em X e Y: regiao central mapeia na tela do totem. "
        "Menor = retangulo ciano maior no video (mais amplitude do dedo para cantos); maior = menos extensao do braco.",
    )
    parser.add_argument("--ema", type=float, default=0.38, help="Peso do frame atual na suavização EMA (0-1).")
    parser.add_argument(
        "--max-step",
        type=float,
        default=120.0,
        help="Máximo de pixels na tela por frame (anti-salto); 0 desliga.",
    )
    parser.add_argument(
        "--inject-mouse",
        action="store_true",
        help="Move o cursor do sistema e injeta cliques (pynput; requer permissões no SO).",
    )
    parser.add_argument("--no-scroll", action="store_true", help="Desativa rolagem por gesto de dois dedos.")
    parser.add_argument("--scroll-invert", action="store_true", help="Inverte o sentido vertical da rolagem.")
    parser.add_argument(
        "--scroll-sensitivity",
        type=float,
        default=None,
        help="Ganho da rolagem (padrão ~18; maior = mais rápido, ex. 24).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Sem janela OpenCV (uso em totem). Encerre com Ctrl+C.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Janela redimensionável em vez de ecrã inteiro (OpenCV).",
    )
    parser.add_argument(
        "--log-events",
        action="store_true",
        help="Com --headless, registra eventos de gesto no stderr.",
    )
    args = parser.parse_args()

    if args.screen_width is not None and args.screen_height is not None:
        sw, sh = args.screen_width, args.screen_height
    else:
        sw, sh = probe_screen_size()

    disp_w, disp_h = probe_screen_size()

    max_step = None if args.max_step <= 0 else args.max_step
    motor_cfg = GestureMotorConfig(
        screen_width=sw,
        screen_height=sh,
        margin_norm=args.margin,
        ema_alpha=args.ema,
        max_step_pixels=max_step,
        scroll_enabled=not args.no_scroll,
        scroll_invert=args.scroll_invert,
        scroll_sensitivity=args.scroll_sensitivity,
    )

    mouse = None
    if args.inject_mouse:
        try:
            from src.mouse_inject import MouseInjector

            mouse = MouseInjector()
        except ImportError:
            print(
                "Instale pynput para --inject-mouse:\n"
                f"  {sys.executable} -m pip install pynput",
                file=sys.stderr,
            )
            return 1

    with CameraCapture(
        device_index=args.camera,
        mirror=not args.no_mirror,
        width=args.width,
        height=args.height,
    ) as cam, GestureMotor(motor_cfg) as motor:
        win = "Motor de Gestos — debug"
        display_fullscreen_done = False
        if not args.headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            if not args.windowed:
                try:
                    cv2.resizeWindow(win, disp_w, disp_h)
                    cv2.moveWindow(win, 0, 0)
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                except cv2.error:
                    pass
        flash_text = ""
        flash_until = 0.0
        prev_scroll_on = False
        event_history: deque[str] = deque(maxlen=14)
        while True:
            ok, frame = cam.read_bgr()
            if not ok or frame is None:
                break
            fh, fw = frame.shape[:2]
            t = time.monotonic()
            rgb = bgr_to_rgb(frame)
            out = motor.process_rgb(rgb, t)
            vis = motor.draw_landmarks(frame, out.results)
            n = out.hand_count
            first_lms = out.results.hand_landmarks[0] if n else None
            tip = out.tip_norm
            cur = out.cursor_screen
            scroll_dy = out.scroll_dy
            scroll_on = out.scroll_active
            events = out.gesture_events

            if args.log_events and events:
                print(t, [e.kind.name for e in events], file=sys.stderr)

            if scroll_dy != 0:
                event_history.append(f"SCROLL {scroll_dy:+d} linhas")

            apply_pynput_mouse(mouse, out)

            for ev in events:
                event_history.append(f"{ev.kind.name} @ ({ev.screen_x},{ev.screen_y})")
                label = {GestureKind.LEFT_CLICK: "CLIQUE ESQ", GestureKind.RIGHT_CLICK: "CLIQUE DIR"}.get(ev.kind)
                if label:
                    flash_text, flash_until = label, t + 0.55
                elif ev.kind == GestureKind.DRAG_START:
                    flash_text, flash_until = "ARRASTAR", t + 0.4
                elif ev.kind == GestureKind.DRAG_END:
                    flash_text, flash_until = "SOLTAR", t + 0.35

            if scroll_on and not prev_scroll_on:
                flash_text, flash_until = "ROLAR (2 dedos)", t + 0.45
            prev_scroll_on = scroll_on

            if tip is not None:
                rx, ry = int(tip[0] * fw), int(tip[1] * fh)
                cv2.circle(vis, (rx, ry), 5, (0, 128, 255), 1, cv2.LINE_AA)
            if cur is not None:
                sx, sy = cur
                px, py = motor.screen_to_preview(sx, sy, fw, fh)
                cv2.circle(vis, (px, py), 14, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(vis, (px, py), 3, (255, 0, 0), -1, cv2.LINE_AA)
                label = f"tela {sx},{sy}"
                cv2.putText(
                    vis,
                    label,
                    (max(5, px - 60), max(25, py - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            touch_lines = None
            if first_lms is not None:
                pr = pinch_ratio(first_lms)
                mr = middle_thumb_pinch_ratio(first_lms)
                pose_scroll = is_scroll_two_finger_pose(first_lms)
                touch_lines = (
                    f"Sinais: i-pinch {pr:.2f}  m-pinch {mr:.2f}",
                    f"Pose scroll 2 dedos: {pose_scroll}  |  scroll ativo: {scroll_on}  dy={scroll_dy}",
                )
            if not args.headless:
                composed = compose_debug_frame(
                    vis,
                    out,
                    motor_cfg,
                    sw,
                    sh,
                    list(event_history),
                    mouse_on=mouse is not None,
                    flash_text=flash_text,
                    t=t,
                    flash_until=flash_until,
                    touch_lines=touch_lines,
                )
                if args.windowed:
                    show = composed
                else:
                    show = _letterbox_to_display(composed, disp_w, disp_h)
                cv2.imshow(win, show)
                if not args.windowed and not display_fullscreen_done:
                    try:
                        cv2.resizeWindow(win, disp_w, disp_h)
                        cv2.moveWindow(win, 0, 0)
                        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    except cv2.error:
                        pass
                    display_fullscreen_done = True
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            else:
                time.sleep(0.002)
    if not args.headless:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
