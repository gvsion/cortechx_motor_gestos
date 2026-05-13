#!/usr/bin/env python3
"""Loop de debug: câmera + MediaPipe Hands + visualização dos landmarks."""

from __future__ import annotations

import argparse
import os
import sys
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
        print(
            "OpenCV (cv2) não está instalado neste interpretador.\n"
            f"  Usado agora: {sys.executable}\n"
            "  Crie o ambiente e instale as dependências, depois rode de novo:\n"
            f"    cd {ROOT}\n"
            "    python3 -m venv .venv\n"
            "    . .venv/bin/activate\n"
            "    pip install -r requirements.txt\n"
            "    python run_debug.py\n"
            "  Ou, se o .venv já existir com tudo instalado:\n"
            f"    {ROOT / '.venv' / 'bin' / 'python3'} run_debug.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

from src.capture import CameraCapture
from src.hand_tracker import HandTracker, HandTrackerConfig, bgr_to_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug: câmera e landmarks das mãos.")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão 0).")
    parser.add_argument("--no-mirror", action="store_true", help="Desativa flip horizontal.")
    parser.add_argument("--width", type=int, default=None, help="Largura solicitada ao driver.")
    parser.add_argument("--height", type=int, default=None, help="Altura solicitada ao driver.")
    args = parser.parse_args()

    cfg = HandTrackerConfig()
    with CameraCapture(
        device_index=args.camera,
        mirror=not args.no_mirror,
        width=args.width,
        height=args.height,
    ) as cam, HandTracker(cfg) as tracker:
        win = "Motor de Gestos — debug"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cam.read_bgr()
            if not ok or frame is None:
                break
            rgb = bgr_to_rgb(frame)
            results = tracker.process(rgb)
            vis = tracker.draw_landmarks(frame, results)
            n = len(results.hand_landmarks) if results.hand_landmarks else 0
            cv2.putText(
                vis,
                f"Maos: {n}  ESC sai",
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
