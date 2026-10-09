"""Самопроверка без экрана: python main.py --selftest итог.txt.

Нужна прежде всего собранному exe: он без консоли, поэтому итог пишется в файл.
Проверяется физика обеих игр (пачка партий со случайными командами: шайба и мяч не
покидают поле сквозь борта и стол), а когда есть окно - партия через окно offscreen.
"""
import os
import tempfile
import time
import traceback


def physics(lines):
    import numpy as np

    from game.core.hockey import P as HP
    from game.core.hockey import Hockey
    from game.core.pong import Pong
    rng = np.random.default_rng(1)
    t0 = time.time()
    h = Hockey(64, seed=1)
    h.pv[:] = rng.normal(size=(64, 2)) * 6
    for _ in range(2400):
        h.step(rng.uniform(-1, 1, (64, 2, 2)))
        x, y = h.puck[:, 0], h.puck[:, 1]
        mouth = (y > HP.goal_lo) & (y < HP.goal_hi)
        if ((y < HP.r - 1e-9) | (y > HP.Y - HP.r + 1e-9)).any() or \
                (~mouth & ((x < HP.r - 1e-9) | (x > HP.X - HP.r + 1e-9))).any():
            raise RuntimeError("шайба вне поля")
    lines.append(f"аэрохоккей: 64 партии по 10 с, голов {int(h.score.sum())}, {time.time() - t0:.2f} с")
    t0 = time.time()
    g = Pong(64, seed=1)
    for _ in range(2400):
        g.step(rng.uniform(-1, 1, (64, 2, 3)))
    lines.append(f"пинг-понг: 64 партии по 10 с, очков {int(g.score.sum())}, {time.time() - t0:.2f} с")


def window(lines):
    from aikit import offscreen
    offscreen.setup(force=True)
    home = tempfile.mkdtemp(prefix="pinpongai-selftest-")
    os.environ["PINPONGAI_HOME"] = home
    from PySide6.QtWidgets import QApplication

    from aikit.host import make_window
    from game import APP_NAME, GAME_INFO, create_widget
    app = QApplication.instance() or QApplication([])
    win, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=home)
    win.show()
    g = win.game
    for game in ("hockey", "pong"):
        t0 = time.time()
        g.change("game", game)
        g.change("mode", "demo")
        g.start_game(seed=3)
        g.simulate(20.0)
        app.processEvents()
        image = win.grab().toImage()
        if image.isNull() or image.width() < 300:
            raise RuntimeError("окно не нарисовалось")
        s = g.match.world.score[0]
        lines.append(f"окно, {game}, ИИ против ИИ 20 с: счёт {s[0]}:{s[1]}, кадр {image.width()}x{image.height()}, "
                     f"{time.time() - t0:.2f} с")
    t0 = time.time()
    win.close()
    app.processEvents()
    lines.append(f"окно закрыто за {time.time() - t0:.2f} с")


def selftest(out_path):
    lines = []
    code = 0
    try:
        import aikit
        from game.version import VERSION
        lines.append(f"PinPongAi {VERSION} самопроверка (aikit {aikit.__version__})")
        lines.append(f"собранная версия: {'да' if aikit.frozen() else 'нет'}")
        physics(lines)
        try:
            import game.ui.widget  # noqa: F401
        except ImportError:
            lines.append("окна ещё нет")
        else:
            window(lines)
        lines.append("ИТОГ: работает")
    except Exception:  # noqa: BLE001 - любая ошибка должна попасть в файл
        lines.append("ИТОГ: ошибка")
        lines.append(traceback.format_exc())
        code = 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return code
