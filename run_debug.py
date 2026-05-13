#!/usr/bin/env python3
"""Loop de debug: câmera + MediaPipe Hands + visualização dos landmarks."""

from __future__ import annotations

import argparse
import os
import sys
import time
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
from src.gesture_interactor import GestureInteractor, GestureInteractorConfig, GestureKind
from src.gesture_math import pinch_ratio
from src.hand_tracker import HandTracker, HandTrackerConfig, bgr_to_rgb
from src.mapping import (
    CursorMapperConfig,
    HandToScreenMapper,
    primary_index_tip_norm,
    probe_screen_size,
)


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
        default=0.15,
        help="Margem normalizada [0,0.49]: região confortável da mão mapeia na tela inteira.",
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
    args = parser.parse_args()

    if args.screen_width is not None and args.screen_height is not None:
        sw, sh = args.screen_width, args.screen_height
    else:
        sw, sh = probe_screen_size()

    max_step = None if args.max_step <= 0 else args.max_step
    map_cfg = CursorMapperConfig(
        screen_width=sw,
        screen_height=sh,
        margin_norm=args.margin,
        ema_alpha=args.ema,
        max_step_pixels=max_step,
    )
    cursor_mapper = HandToScreenMapper(map_cfg)
    gesture_cfg = GestureInteractorConfig()
    gestures = GestureInteractor(gesture_cfg)

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

    cfg = HandTrackerConfig()
    with CameraCapture(
        device_index=args.camera,
        mirror=not args.no_mirror,
        width=args.width,
        height=args.height,
    ) as cam, HandTracker(cfg) as tracker:
        win = "Motor de Gestos — debug"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        flash_text = ""
        flash_until = 0.0
        while True:
            ok, frame = cam.read_bgr()
            if not ok or frame is None:
                break
            fh, fw = frame.shape[:2]
            rgb = bgr_to_rgb(frame)
            results = tracker.process(rgb)
            vis = tracker.draw_landmarks(frame, results)
            n = len(results.hand_landmarks) if results.hand_landmarks else 0
            first_lms = results.hand_landmarks[0] if n else None
            tip = primary_index_tip_norm(results.hand_landmarks)
            cur = cursor_mapper.update(tip)
            t = time.monotonic()
            events = gestures.update(t, first_lms, cur)

            if mouse is not None and cur is not None:
                mouse.move(cur[0], cur[1])
            for ev in events:
                if mouse is not None:
                    if ev.kind == GestureKind.LEFT_CLICK:
                        mouse.left_click()
                    elif ev.kind == GestureKind.RIGHT_CLICK:
                        mouse.right_click()
                    elif ev.kind == GestureKind.DRAG_START:
                        mouse.left_down()
                    elif ev.kind == GestureKind.DRAG_END:
                        mouse.left_up()
                label = {GestureKind.LEFT_CLICK: "CLIQUE ESQ", GestureKind.RIGHT_CLICK: "CLIQUE DIR"}.get(ev.kind)
                if label:
                    flash_text, flash_until = label, t + 0.55
                elif ev.kind == GestureKind.DRAG_START:
                    flash_text, flash_until = "ARRASTAR", t + 0.4
                elif ev.kind == GestureKind.DRAG_END:
                    flash_text, flash_until = "SOLTAR", t + 0.35

            if tip is not None:
                rx, ry = int(tip[0] * fw), int(tip[1] * fh)
                cv2.circle(vis, (rx, ry), 5, (0, 128, 255), 1, cv2.LINE_AA)
            if cur is not None:
                sx, sy = cur
                px, py = cursor_mapper.screen_to_preview_frame(sx, sy, fw, fh)
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
            if first_lms is not None:
                pr = pinch_ratio(first_lms)
                cv2.putText(
                    vis,
                    f"pinch {pr:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )
            if t < flash_until and flash_text:
                cv2.putText(
                    vis,
                    flash_text,
                    (fw // 2 - 80, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                vis,
                f"Maos: {n}  tela {sw}x{sh}  mouse={'on' if mouse else 'off'}  ESC sai",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
