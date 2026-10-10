"""Кадры для README: окно без экрана, обе игры в режиме «ИИ против ИИ» с «Мыслями ИИ».

    python tools/make_previews.py [папка]     # по умолчанию docs/
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aikit import offscreen  # noqa: E402

offscreen.setup(force=True)
os.environ["PINPONGAI_HOME"] = tempfile.mkdtemp(prefix="pinpongai-previews-")


def rally_frame(g, game, limit=40.0):
    """Прожить партию до момента, когда мяч летит и ИИ уже рассчитал удар."""
    w = g.match.world
    t = 0.0
    while t < limit:
        g.simulate(0.05)
        t += 0.05
        if t < 2.0:
            continue
        if game == "pong":
            if w.phase[0] == 1 and w.last_hitter[0] >= 0 and 0.4 < w.ball[0, 0] < 2.3 and w.ball[0, 1] > 0.12:
                return
        else:
            if abs(w.pv[0, 0]) > 1.5 and 0.5 < w.puck[0, 0] < 1.5:
                return


def main(out):
    from PySide6.QtWidgets import QApplication

    from aikit.host import make_window
    from game import APP_NAME, GAME_INFO, create_widget

    os.makedirs(out, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    window, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=os.environ["PINPONGAI_HOME"])
    window.resize(1280, 800)
    window.show()
    g = window.game
    shots = []
    for theme_name in ("dark", "light"):
        ctx.set_theme(theme_name)
        suffix = "" if theme_name == "dark" else "_light"
        g.show_page("menu")
        app.processEvents()
        shots.append((window.grab(), f"preview_menu{suffix}.png"))
        g.change("show_thoughts", True)
        g.change("mode", "demo")
        g.change("level", "strong")
        for game in ("hockey", "pong"):
            g.change("game", game)
            g.start_game(seed=11)
            g.timer.stop()
            rally_frame(g, game)
            app.processEvents()
            shots.append((window.grab(), f"preview_{game}{suffix}.png"))
        g.show_page("stats")
        app.processEvents()
        shots.append((window.grab(), f"preview_stats{suffix}.png"))
    ctx.set_theme("dark")
    window.resize(1280, 720)
    g.change("game", "pong")
    g.start_game(seed=11)
    g.timer.stop()
    rally_frame(g, "pong")
    app.processEvents()
    shots.append((window.grab(), "preview.png"))
    g.change("show_thoughts", False)
    g.change("mode", "ai")
    for pix, name in shots:
        pix.save(os.path.join(out, name))
        print(name)
    g.shutdown()
    window.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs"))
