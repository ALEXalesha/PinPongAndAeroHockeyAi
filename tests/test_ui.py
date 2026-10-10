"""Окно без экрана (Qt offscreen): обе игры во всех режимах, мышь без захвата, колесо и
клавиши, «Мысли ИИ», темы и языки, статистика, закрытие за 2 секунды. Мышь и клавиши -
собранными событиями, настоящий курсор не трогается."""
import os
import re
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from game import APP_NAME, GAME_INFO, create_widget
from game.ui.i18n import EN, RU

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path):
    from aikit.host import make_window
    w, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=str(tmp_path / "home"))
    w.resize(1280, 800)
    w.show()
    app.processEvents()
    yield w
    w.game.shutdown()
    w.close()


def start(win, game, mode, level="medium"):
    g = win.game
    g.change("game", game)
    g.change("mode", mode)
    g.change("level", level)
    g.start_game(seed=5)
    g.timer.stop()                    # время ведёт тест, а не таймер
    QApplication.processEvents()
    return g


def move_mouse(widget, x, y):
    ev = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y), Qt.MouseButton.NoButton,
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)


def test_translations_complete():
    assert set(RU) == set(EN)


@pytest.mark.parametrize("game", ["hockey", "pong"])
@pytest.mark.parametrize("mode", ["ai", "duo", "demo"])
def test_every_game_and_mode_runs_and_draws(app, win, game, mode):
    g = start(win, game, mode)
    g.simulate(3.0)
    img = win.grab().toImage()
    assert not img.isNull() and img.width() >= 900
    assert g.match.world.ticks == 720
    assert g.play.score.text()


def test_mouse_moves_mallet_without_grabbing(app, win):
    g = start(win, "hockey", "ai")
    f = g.play.field
    sx, sy = f.to_screen(0.6, 0.25).x(), f.to_screen(0.6, 0.25).y()
    move_mouse(f, sx, sy)
    assert QWidget.mouseGrabber() is None
    g.simulate(1.0)
    assert np.linalg.norm(g.match.world.mal[0, 0] - [0.6, 0.25]) < 0.02
    # курсор на чужой половине - бита у центральной линии, не дальше
    p = f.to_screen(1.6, 0.5)
    move_mouse(f, p.x(), p.y())
    g.simulate(1.0)
    assert g.match.world.mal[0, 0, 0] <= 1.0 - 0.05 + 1e-9


def test_no_cursor_capture_anywhere_in_code():
    """Ни захвата мыши, ни перемещения или скрытия курсора программой - во всём game/."""
    bad = re.compile(r"grabMouse|QCursor\.setPos|setPos\(|ClipCursor|BlankCursor|grabKeyboard")
    found = []
    for base, _, files in os.walk(os.path.join(ROOT, "game")):
        for name in files:
            if name.endswith(".py"):
                text = open(os.path.join(base, name), encoding="utf-8").read()
                if bad.search(text):
                    found.append(name)
    assert not found


def test_wheel_tilts_paddle(app, win):
    g = start(win, "pong", "ai")
    f = g.play.field
    h = g.match.humans[0]
    before = h.tilt
    ev = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(f, ev)
    assert abs(h.tilt - (before + 0.2)) < 1e-9
    g.simulate(0.5)
    assert abs(g.match.world.th[0, 0] - h.tilt) < 0.01


def test_second_player_on_keys(app, win):
    g = start(win, "hockey", "duo")
    f = g.play.field
    h = g.match.humans[1]
    y0 = h.target[1]
    QApplication.sendEvent(f, QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier))
    g.simulate(0.3)
    QApplication.sendEvent(f, QKeyEvent(QKeyEvent.Type.KeyRelease, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier))
    assert h.target[1] < y0 - 0.3                 # вверх на экране - к меньшему y
    y1 = h.target[1]
    g.simulate(0.3)
    assert h.target[1] == y1


def test_thoughts_draw_for_ai(app, win):
    g = start(win, "pong", "demo")
    g.change("show_thoughts", True)
    g.simulate(1.5)
    got = [g.match.thoughts(s) for s in (0, 1)]
    assert any(x is not None for x in got)
    assert not win.grab().toImage().isNull()
    g.change("show_thoughts", False)


def test_ai_win_is_recorded_once(app, win):
    g = start(win, "hockey", "ai", "strong")
    g.match.world.score[0] = [0, 6]
    for _ in range(60):
        g.simulate(1.0)
        if g.match.world.over[0]:
            break
    assert g.match.world.over[0]
    e = g.ctx.stats.entry("hockey/strong")
    assert e["games"] == 1
    g.simulate(1.0)
    assert g.ctx.stats.entry("hockey/strong")["games"] == 1
    assert g.play.status.text()


def test_pause_stops_time(app, win):
    g = start(win, "hockey", "demo")
    g.start_timer()
    g.toggle_pause()
    assert g.paused and not g.timer.isActive()
    g.toggle_pause()
    assert g.timer.isActive()


def test_theme_and_language_switch(app, win):
    g = start(win, "pong", "ai")
    for th in ("light", "dark"):
        g.ctx.set_theme(th)
        assert g.play.field.theme == th
        assert not win.grab().toImage().isNull()
    g.ctx.set_lang("en")
    assert g.play.menu_button.text() == EN["menu"]
    g.ctx.set_lang("ru")
    assert g.play.menu_button.text() == RU["menu"]


def test_pages_open(app, win):
    g = win.game
    for page in ("stats", "settings", "menu"):
        g.show_page(page)
        assert g.current_page() == page
        assert not win.grab().toImage().isNull()


def test_strict_serve_only_for_people(app, win):
    g = win.game
    g.change("strict_serve", True)
    start(win, "pong", "ai")
    assert g.match.world.strict[0].tolist() == [True, False]
    g.change("strict_serve", False)


def test_close_within_two_seconds_and_window_remembered(app, tmp_path):
    from aikit.host import make_window
    root = str(tmp_path / "home2")
    w, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=root)
    w.show()
    w.game.change("mode", "demo")
    w.game.start_game(seed=1)
    t0 = time.time()
    w.close()
    app.processEvents()
    assert time.time() - t0 < 2.0
    assert not w.game.timer.isActive()
    assert os.path.exists(os.path.join(root, "window.json"))
