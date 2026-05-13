"""Injeção opcional de mouse via pynput (Linux/X11 ou Wayland conforme suporte)."""

from __future__ import annotations


class MouseInjector:
    """Move o cursor e cliques esquerdo/direito."""

    def __init__(self) -> None:
        from pynput.mouse import Button, Controller

        self._c = Controller()
        self._left = Button.left
        self._right = Button.right

    def move(self, x: int, y: int) -> None:
        self._c.position = (int(x), int(y))

    def left_click(self) -> None:
        self._c.click(self._left, 1)

    def left_down(self) -> None:
        self._c.press(self._left)

    def left_up(self) -> None:
        self._c.release(self._left)

    def right_click(self) -> None:
        self._c.click(self._right, 1)

    def scroll_vertical(self, dy: int) -> None:
        """dy em passos da roda (pynput); sinal depende do SO / driver."""
        if dy == 0:
            return
        self._c.scroll(0, int(dy))
