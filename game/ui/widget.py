"""Виджет игры (create_widget по контракту): меню, поле, статистика, настройки.

Окно даёт оболочка (aikit.run_standalone или ИИ-Игротека). Тема и язык приходят из ctx и
меняются на лету по ctx.changed. Файлы - только в ctx.data_dir. Физика идёт по часам
(game/core/control.Clock): таймер окна лишь сообщает, сколько прошло времени.
"""
import json
import os
import random

from PySide6.QtCore import QElapsedTimer, QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                               QScrollArea, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from aikit import settings as st
from aikit import theme
from aikit.widgets import Combo, button, card, label, restyle, segmented
from game.ai.match import model_path
from game.core import hockey, pong
from game.ui.field import Field
from game.ui.i18n import translator
from game.ui.session import MatchSession
from game.version import VERSION

GAMES = ("hockey", "pong")
LEVELS = ("novice", "medium", "strong")
MODES = ("ai", "duo", "demo")
SCHEMA = {
    "game": ("hockey", st.choice(*GAMES)),
    "level": ("medium", st.choice(*LEVELS)),
    "mode": ("ai", st.choice(*MODES)),
    "show_thoughts": (False, st.boolean()),
    "strict_serve": (False, st.boolean()),
}
FRAME_MS = 15


def page_widget():
    w = QWidget()
    w.setObjectName("page")
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return w


def fmt_time(t, seconds):
    if seconds < 90:
        return t("seconds", n=round(seconds))
    if seconds < 5400:
        return t("minutes", n=round(seconds / 60))
    return t("hours", n=f"{seconds / 3600:.1f}")


def number(n):
    return f"{n:,}".replace(",", " ")


def to_win(game):
    return hockey.P.to_win if game == "hockey" else pong.P.to_win


# --- меню ---


class MenuPage(QWidget):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.addStretch(2)
        col = QVBoxLayout()
        col.setSpacing(10)
        col.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QLabel

        from game.ui.icon import render
        self.logo = QLabel()
        self.logo.setPixmap(QPixmap.fromImage(render(112)))
        col.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignHCenter)
        self.title = label("", "title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self.title)
        self.sub = label("", "muted", wrap=True)
        self.sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub.setFixedWidth(620)
        col.addWidget(self.sub, 0, Qt.AlignmentFlag.AlignHCenter)
        col.addSpacing(12)
        box = card()
        box.setMaximumWidth(640)
        grid = QGridLayout(box)
        grid.setContentsMargins(22, 18, 22, 18)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        self.captions = {}

        def row(r, key, widget):
            cap = label("", "muted")
            self.captions[key] = cap
            grid.addWidget(cap, r, 0)
            grid.addWidget(widget, r, 1)

        w, self.game_buttons = segmented({g: "" for g in GAMES}, lambda v: gw.change("game", v))
        row(0, "game", w)
        w, self.level_buttons = segmented({lv: "" for lv in LEVELS}, lambda v: gw.change("level", v))
        row(1, "level", w)
        w, self.mode_buttons = segmented({m: "" for m in MODES}, lambda v: gw.change("mode", v))
        row(2, "mode", w)
        col.addWidget(box, 0, Qt.AlignmentFlag.AlignHCenter)
        self.note = label("", "hint", wrap=True)
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note.setFixedWidth(620)
        col.addWidget(self.note, 0, Qt.AlignmentFlag.AlignHCenter)
        col.addSpacing(8)
        self.play_button = button("", "menuPrimary", lambda: gw.start_game())
        self.stats_button = button("", "menu", lambda: gw.show_page("stats"))
        self.settings_button = button("", "menu", lambda: gw.show_page("settings"))
        for b in (self.play_button, self.stats_button, self.settings_button):
            col.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addLayout(col)
        outer.addStretch(3)
        self.foot = label("", "hint")
        self.foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.foot)

    def sync(self, s):
        self.game_buttons[s["game"]].setChecked(True)
        self.level_buttons[s["level"]].setChecked(True)
        self.mode_buttons[s["mode"]].setChecked(True)
        trained = os.path.exists(model_path(s["game"]))
        self.note.setText("" if trained else self.gw.t("no_net"))
        self.note.setVisible(not trained)

    def retranslate(self):
        t = self.gw.t
        self.title.setText(t("app_title"))
        self.sub.setText(t("app_sub"))
        for key, cap in self.captions.items():
            cap.setText(t(key))
        for k, b in self.game_buttons.items():
            b.setText(t(k))
        for k, b in self.level_buttons.items():
            b.setText(t(k))
        for k, b in self.mode_buttons.items():
            b.setText(t("mode_" + k))
        self.play_button.setText(t("play"))
        self.stats_button.setText(t("stats"))
        self.settings_button.setText(t("settings"))
        self.foot.setText(t("footer", v=VERSION))
        self.sync(self.gw.settings)


# --- партия ---


class PlayPage(QWidget):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        self.field = Field()
        root.addWidget(self.field, 1)
        side = card()
        side.setFixedWidth(300)
        col = QVBoxLayout(side)
        col.setContentsMargins(16, 14, 16, 14)
        col.setSpacing(8)
        self.heading = label("", "section", wrap=True)
        col.addWidget(self.heading)
        self.score = label("", "bigNumber")
        col.addWidget(self.score)
        self.status = label("", "status", wrap=True)
        col.addWidget(self.status)
        self.controls = label("", "hint", wrap=True)
        col.addWidget(self.controls)
        self.thoughts_box = QCheckBox("")
        self.thoughts_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.thoughts_box.toggled.connect(lambda on: gw.change("show_thoughts", on))
        col.addWidget(self.thoughts_box)
        self.thoughts_hint = label("", "hint", wrap=True)
        col.addWidget(self.thoughts_hint)
        self.pause_button = button("", None, gw.toggle_pause)
        col.addWidget(self.pause_button)
        col.addStretch(1)
        last = QHBoxLayout()
        self.again_button = button("", "primary", lambda: gw.start_game())
        self.menu_button = button("", None, lambda: gw.show_page("menu"))
        last.addWidget(self.again_button, 1)
        last.addWidget(self.menu_button, 1)
        col.addLayout(last)
        root.addWidget(side)
        self.side = side

    def retranslate(self):
        t = self.gw.t
        self.thoughts_box.setText(t("thoughts"))
        self.thoughts_hint.setText(t("thoughts_hint"))
        self.again_button.setText(t("again"))
        self.menu_button.setText(t("menu"))
        self.pause_button.setText(t("resume") if self.gw.paused else t("pause"))


# --- статистика ---


def table_widget(cols):
    tw = QTableWidget(0, cols)
    tw.verticalHeader().setVisible(False)
    tw.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    tw.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    tw.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    tw.setShowGrid(False)
    tw.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    tw.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    header = tw.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    header.setHighlightSections(False)
    return tw


def fit_height(tw):
    tw.resizeRowsToContents()
    tw.setFixedHeight(tw.horizontalHeader().height() + sum(tw.rowHeight(r) for r in range(tw.rowCount())) + 4)


def item(text, align=Qt.AlignmentFlag.AlignCenter):
    it = QTableWidgetItem(text)
    it.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
    return it


class StatsPage(QWidget):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        body = page_widget()
        scroll.setWidget(body)
        col = QVBoxLayout(body)
        col.setContentsMargins(26, 22, 26, 22)
        col.setSpacing(14)
        top = QHBoxLayout()
        self.heading = label("", "heading")
        top.addWidget(self.heading)
        top.addStretch(1)
        self.menu_button = button("", None, lambda: gw.show_page("menu"))
        top.addWidget(self.menu_button)
        col.addLayout(top)
        mine = card()
        mc = QVBoxLayout(mine)
        mc.setContentsMargins(18, 14, 18, 14)
        self.mine_title = label("", "section")
        mc.addWidget(self.mine_title)
        self.table = table_widget(5)
        mc.addWidget(self.table)
        col.addWidget(mine)
        self.passports = {}
        for g in GAMES:
            pc = card()
            lay = QVBoxLayout(pc)
            lay.setContentsMargins(18, 14, 18, 14)
            title = label("", "section")
            text = label("", "muted", wrap=True)
            checks = table_widget(3)
            foot = label("", "hint", wrap=True)
            for w in (title, text, checks, foot):
                lay.addWidget(w)
            col.addWidget(pc)
            self.passports[g] = (title, text, checks, foot)
        col.addStretch(1)

    def refresh(self):
        t = self.gw.t
        self.heading.setText(t("stats_title"))
        self.menu_button.setText(t("menu"))
        self.mine_title.setText(t("your_games"))
        self.table.setHorizontalHeaderLabels([t(k) for k in ("col_setup", "col_games", "col_wins", "col_losses",
                                                               "col_streak")])
        rows = [(g, lv) for g in GAMES for lv in LEVELS]
        self.table.setRowCount(len(rows))
        stats = self.gw.ctx.stats
        for r, (g, lv) in enumerate(rows):
            e = stats.entry(f"{g}/{lv}")
            self.table.setItem(r, 0, item(f"{t(g)} · {t(lv)}", Qt.AlignmentFlag.AlignLeft))
            for c, f in enumerate(("games", "wins", "losses", "best_streak"), start=1):
                self.table.setItem(r, c, item(str(e[f])))
        fit_height(self.table)
        for g, (title, text, checks, foot) in self.passports.items():
            title.setText(f"{t('passport')}: {t(g)}")
            try:
                with open(model_path(g, "json"), encoding="utf-8") as f:
                    pp = json.load(f)
            except (OSError, ValueError):
                text.setText(t("not_trained"))
                checks.hide()
                foot.hide()
                continue
            net, tr = pp["network"], pp["training"]
            text.setText(t("passport_text", features=net["features"], hidden=" × ".join(map(str, net["hidden"])),
                           params=number(net["parameters"]), imitation=number(tr.get("imitation", 0)),
                           gens=tr["gens"], games=number(tr["games"]), time=fmt_time(t, tr["seconds"])))
            checks.setHorizontalHeaderLabels([t("col_check"), t("col_result"), t("col_points")])
            rows = pp["checks"]
            checks.setRowCount(len(rows))
            for r, ch in enumerate(rows):
                name = t(ch["subject"])
                opp = t(ch["opponent"])
                checks.setItem(r, 0, item(f"{name} {t('vs')} {opp}", Qt.AlignmentFlag.AlignLeft))
                checks.setItem(r, 1, item(f"{100 * ch['win_rate']:.1f}%"))
                checks.setItem(r, 2, item(f"{ch['points_for']:.1f} : {ch['points_against']:.1f}"))
            fit_height(checks)
            checks.show()
            foot.setText(t("checks_text", n=pp.get("games_per_check", 200)))
            foot.show()


# --- настройки ---


class SettingsPage(QWidget):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        outer0 = QVBoxLayout(self)
        outer0.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer0.addWidget(scroll)
        body = page_widget()
        scroll.setWidget(body)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(26, 22, 26, 22)
        outer.setSpacing(14)
        top = QHBoxLayout()
        self.heading = label("", "heading")
        top.addWidget(self.heading)
        top.addStretch(1)
        self.menu_button = button("", None, lambda: gw.show_page("menu"))
        top.addWidget(self.menu_button)
        outer.addLayout(top)
        box = card()
        box.setMaximumWidth(760)
        self.grid = QGridLayout(box)
        self.grid.setContentsMargins(22, 18, 22, 18)
        self.grid.setHorizontalSpacing(22)
        self.grid.setVerticalSpacing(16)
        self.grid.setColumnStretch(0, 1)
        self.rows = {}
        self.theme_combo = Combo()
        for k in theme.THEMES:
            self.theme_combo.addItem("", k)
        self.theme_combo.currentIndexChanged.connect(lambda i: gw.ctx.set_theme(self.theme_combo.itemData(i)))
        self._row("theme", self.theme_combo)
        self.lang_combo = Combo()
        self.lang_combo.addItem("Русский", "ru")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.currentIndexChanged.connect(lambda i: gw.ctx.set_lang(self.lang_combo.itemData(i)))
        self._row("language", self.lang_combo)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(gw.ctx.sound.volume)
        self.volume.valueChanged.connect(gw.ctx.set_volume)
        self._row("volume", self.volume)
        self.strict_box = QCheckBox("")
        self.strict_box.toggled.connect(lambda on: gw.change("strict_serve", on))
        self._row("strict_serve", self.strict_box)
        outer.addWidget(box)
        self.note = label("", "hint")
        outer.addWidget(self.note)
        outer.addStretch(1)

    def _row(self, key, widget):
        r = len(self.rows)
        names = QVBoxLayout()
        names.setSpacing(2)
        title = label("", "section")
        hint = label("", "hint", wrap=True)
        names.addWidget(title)
        names.addWidget(hint)
        self.grid.addLayout(names, r, 0)
        self.grid.addWidget(widget, r, 1)
        self.rows[key] = (title, hint)

    def retranslate(self):
        t = self.gw.t
        self.heading.setText(t("settings_title"))
        self.menu_button.setText(t("menu"))
        for key, (title, hint) in self.rows.items():
            title.setText(t(key))
            hint.setText(t(key + "_hint"))
        for i, k in enumerate(theme.THEMES):
            self.theme_combo.setItemText(i, t(k))
        self.note.setText(t("saved_note"))
        self.sync()

    def sync(self):
        g = self.gw
        for combo, value in ((self.theme_combo, g.ctx.theme), (self.lang_combo, g.ctx.lang)):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(False)
        self.strict_box.blockSignals(True)
        self.strict_box.setChecked(g.settings["strict_serve"])
        self.strict_box.blockSignals(False)


# --- сам виджет игры ---


class GameWidget(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.t = translator(ctx.lang)
        self.settings = st.Settings(SCHEMA, ctx.file("settings.json"))
        self.match = None
        self.paused = False
        self.recorded = False
        self.rng = random.Random()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.on_frame)
        self.elapsed = QElapsedTimer()
        self.stack = QStackedWidget(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.stack)
        self.menu = MenuPage(self)
        self.play = PlayPage(self)
        self.stats = StatsPage(self)
        self.settings_page = SettingsPage(self)
        self.pages = {"menu": self.menu, "play": self.play, "stats": self.stats, "settings": self.settings_page}
        for p in self.pages.values():
            self.stack.addWidget(p)
        for keys, slot in (("Escape", lambda: self.show_page("menu")), ("Space", self.toggle_pause),
                           ("N", lambda: self.start_game())):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        ctx.changed.connect(self.on_ctx_changed)
        self.apply_theme()
        self.retranslate()
        self.show_page("menu")

    def minimumSizeHint(self):
        return QSize(940, 600)

    # --- общее ---

    def current_page(self):
        w = self.stack.currentWidget()
        return next(k for k, p in self.pages.items() if p is w)

    def show_page(self, name):
        if name != "play":
            self.timer.stop()
        self.stack.setCurrentWidget(self.pages[name])
        if name == "stats":
            self.stats.refresh()
        elif name == "settings":
            self.settings_page.sync()
        elif name == "menu":
            self.menu.sync(self.settings)
        elif name == "play" and self.match is not None and not self.paused:
            self.start_timer()

    def change(self, key, value):
        if self.settings[key] == value:
            return
        self.settings.set(key, value)
        if key in ("game", "level", "mode"):
            self.menu.sync(self.settings)
        elif key == "show_thoughts":
            self.play.field.show_thoughts = value
            self.play.thoughts_hint.setVisible(value)
            self.play.field.update()

    def on_ctx_changed(self, what):
        if what == "theme":
            self.apply_theme()
        elif what == "lang":
            self.t = translator(self.ctx.lang)
            self.retranslate()

    def apply_theme(self):
        c = theme.palette(self.ctx.theme)
        self.play.field.set_theme(self.ctx.theme)
        for combo in self.findChildren(Combo):
            combo.set_arrow_color(c["text"])
        self.update()

    def retranslate(self):
        self.menu.retranslate()
        self.play.retranslate()
        self.settings_page.retranslate()
        if self.current_page() == "stats":
            self.stats.refresh()
        self.refresh()

    def shutdown(self):
        """Окно закрывается: таймер стоп. Потоков у игры нет - ИИ считается в шаге физики."""
        self.timer.stop()
        return True

    # --- партия ---

    def sides(self):
        mode = self.settings["mode"]
        if mode == "ai":
            return ("mouse", "ai")
        if mode == "duo":
            return ("mouse", "keys")
        return ("ai", "ai")

    def start_game(self, seed=None):
        s = self.settings
        if seed is None:
            seed = self.rng.randrange(2 ** 31)
        self.match = MatchSession(s["game"], self.sides(), s["level"], seed=seed, first=seed % 2,
                                  strict_serve=s["strict_serve"])
        self.recorded = False
        self.paused = False
        self.play.field.banner = ""
        self.play.field.set_session(self.match)
        self.play.field.show_thoughts = s["show_thoughts"]
        self.stack.setCurrentWidget(self.play)
        self.play.field.setFocus()
        self.start_timer()
        self.refresh()

    def start_timer(self):
        self.elapsed.restart()
        self.timer.start(FRAME_MS)

    def toggle_pause(self):
        if self.match is None or self.current_page() != "play":
            return
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.play.field.banner = self.t("paused")
        else:
            self.play.field.banner = ""
            self.start_timer()
        self.play.pause_button.setText(self.t("resume") if self.paused else self.t("pause"))
        self.play.field.update()

    def on_frame(self):
        if self.match is None:
            return
        dt = self.elapsed.restart() / 1000.0
        self.match.advance(dt)
        self.handle_events()
        self.play.field.update()

    def simulate(self, seconds):
        """Без таймера: прожить seconds игрового времени (проверки и самопроверка)."""
        self.match.simulate(seconds)
        self.handle_events()
        self.play.field.update()

    def handle_events(self):
        changed = False
        for ev in self.match.take_events():
            if ev == "hit":
                self.ctx.sound.play("move")
            elif isinstance(ev, tuple) and ev[0] in ("goal", "point"):
                self.ctx.sound.play("capture")
                changed = True
            elif isinstance(ev, tuple) and ev[0] == "over":
                self.finish(ev[1])
                changed = True
        if changed:
            self.refresh()

    def finish(self, winner):
        self.timer.stop()
        if self.recorded:
            return
        self.recorded = True
        mode = self.settings["mode"]
        if mode == "ai":
            result = "win" if winner == 0 else "loss"
            self.ctx.stats.record(f"{self.match.game}/{self.match.level}", result)
            self.ctx.sound.play(result)
        else:
            self.ctx.sound.play("win")
        self.refresh()

    def outcome_text(self, winner):
        mode = self.settings["mode"]
        if mode == "ai":
            return self.t("win") if winner == 0 else self.t("loss")
        return self.t("left_won") if winner == 0 else self.t("right_won")

    def names(self):
        mode = self.settings["mode"]
        if mode == "ai":
            return self.t("you"), self.t("ai")
        if mode == "duo":
            return self.t("you"), self.t("player2")
        return self.t("ai"), self.t("ai")

    def refresh(self):
        m = self.match
        pp = self.play
        if m is None:
            return
        t = self.t
        w = m.world
        left, right = self.names()
        pp.heading.setText(f"{t(m.game)} · {t(m.level)} · {t('score_to', n=to_win(m.game))}")
        a, b = w.score[0]
        pp.score.setText(f"{left} {a} : {b} {right}")
        if w.over[0]:
            win = int(w.winner[0])
            restyle(pp.status, "statusWin" if (self.settings["mode"] != "ai" or win == 0) else "statusLoss")
            pp.status.setText(self.outcome_text(win))
            pp.field.banner = self.outcome_text(win)
        else:
            restyle(pp.status, "status")
            if m.game == "pong":
                srv = int(w.server[0])
                kind = m.sides[srv]
                key = {"mouse": "serve_you", "keys": "serve_p2", "ai": "serve_ai"}[kind]
                if self.settings["mode"] == "demo":
                    key = "serve_left" if srv == 0 else "serve_right"
                pp.status.setText(t(key))
            else:
                pp.status.setText("")
        ctl = t("controls_" + m.game)
        if self.settings["mode"] == "duo":
            ctl += " " + t("controls_duo_" + m.game)
        if self.settings["mode"] == "demo":
            ctl = ""
        pp.controls.setText(ctl)
        pp.controls.setVisible(bool(ctl))
        pp.thoughts_box.blockSignals(True)
        pp.thoughts_box.setChecked(self.settings["show_thoughts"])
        pp.thoughts_box.blockSignals(False)
        pp.thoughts_hint.setVisible(self.settings["show_thoughts"])
        pp.pause_button.setText(t("resume") if self.paused else t("pause"))
