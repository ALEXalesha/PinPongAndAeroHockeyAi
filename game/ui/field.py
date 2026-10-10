"""Поле: рисует партию и принимает мышь и клавиши.

Курсор не захватывается: ни захвата мыши окном, ни перемещения курсора программой, ни скрытия.
Мышь только сообщает точку (setMouseTracking), а бита или ракетка едет к ней с пределами
скорости и ускорения. Ушёл курсор с поля - бита просто стоит у края.
"""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from aikit import theme
from game.core import pong
from game.ui.session import TILT_STEP

KEYMAP = {Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right", Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
          Qt.Key.Key_PageUp: "tilt+", Qt.Key.Key_PageDown: "tilt-"}

COLORS = {
    "dark": {"rink": "#25384d", "rink_line": "#c9d6e6", "center": "#d8404f", "goal": "#11161e",
             "table": "#1d4f86", "table_edge": "#e9eef5", "net": "#c9d1dd", "floor": "#1a1f28",
             "puck": "#0d1015", "puck_edge": "#e6e8ec", "ball": "#ff9f1c",
             "left": "#e0525f", "right": "#4f8ff0", "handle": "#c8a27a", "path": "#ffd166", "aim": "#ffd166"},
    "light": {"rink": "#eef4fa", "rink_line": "#8ba0c4", "center": "#d8404f", "goal": "#2c3140",
              "table": "#2a6db8", "table_edge": "#ffffff", "net": "#5f6575", "floor": "#d5dbe5",
              "puck": "#1b2030", "puck_edge": "#1b2030", "ball": "#f08a00",
              "left": "#d8404f", "right": "#3a6fd8", "handle": "#a7764a", "path": "#c47f00", "aim": "#c47f00"},
}

PONG_VIEW = (-1.15, pong.P.L + 1.15, pong.P.floor, 1.0)     # xmin, xmax, ymin, ymax


class Field(QWidget):
    paused_toggled = Signal()

    def __init__(self):
        super().__init__()
        self.session = None
        self.theme = "dark"
        self.show_thoughts = False
        self.banner = ""
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(560, 360)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_session(self, session):
        self.session = session
        self.update()

    def set_theme(self, name):
        self.theme = name
        self.update()

    # --- система координат ---

    def frame(self):
        """(прямоугольник мира на экране, xmin, xmax, ymin, ymax, y вниз?)"""
        m = 16
        W, H = self.width() - 2 * m, self.height() - 2 * m
        if self.session is not None and self.session.game == "pong":
            x0, x1, y0, y1 = PONG_VIEW
            down = False
        else:
            x0, x1, y0, y1 = -0.08, 2.08, -0.04, 1.04
            down = True
        aspect = (x1 - x0) / (y1 - y0)
        w = min(W, H * aspect)
        h = w / aspect
        rect = QRectF(m + (W - w) / 2, m + (H - h) / 2, w, h)
        return rect, x0, x1, y0, y1, down

    def to_screen(self, x, y):
        rect, x0, x1, y0, y1, down = self.frame()
        sx = rect.left() + (x - x0) / (x1 - x0) * rect.width()
        t = (y - y0) / (y1 - y0)
        sy = rect.top() + (t if down else 1 - t) * rect.height()
        return QPointF(sx, sy)

    def to_world(self, px, py):
        rect, x0, x1, y0, y1, down = self.frame()
        x = x0 + (px - rect.left()) / max(rect.width(), 1) * (x1 - x0)
        t = (py - rect.top()) / max(rect.height(), 1)
        y = y0 + (t if down else 1 - t) * (y1 - y0)
        return x, y

    def scale(self):
        rect, x0, x1, *_ = self.frame()
        return rect.width() / (x1 - x0)

    # --- ввод ---

    def mouse_human(self):
        s = self.session
        if s is None:
            return None
        for side, kind in enumerate(s.sides):
            if kind == "mouse":
                return s.humans[side]
        return None

    def key_human(self):
        s = self.session
        if s is None:
            return None
        for side, kind in enumerate(s.sides):
            if kind == "keys":
                return s.humans[side]
        return None

    def mouseMoveEvent(self, event):
        h = self.mouse_human()
        if h is not None:
            x, y = self.to_world(event.position().x(), event.position().y())
            h.target[:] = (x, y)
            h.clamp(self.session.world)
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        h = self.mouse_human()
        if h is not None and self.session.game == "pong":
            steps = event.angleDelta().y() / 120
            h.tilt += steps * TILT_STEP
            h.clamp(self.session.world)
            self.update()
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        h = self.key_human()
        k = KEYMAP.get(event.key())
        if h is not None and k is not None:
            if not event.isAutoRepeat():
                h.keys.add(k)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        h = self.key_human()
        k = KEYMAP.get(event.key())
        if h is not None and k is not None:
            if not event.isAutoRepeat():
                h.keys.discard(k)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        h = self.key_human()
        if h is not None:
            h.keys.clear()
        super().focusOutEvent(event)

    # --- рисование ---

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = theme.palette(self.theme)
        p.fillRect(self.rect(), QColor(pal["bg"]))
        s = self.session
        if s is not None:
            if s.game == "hockey":
                self.paint_hockey(p, s)
            else:
                self.paint_pong(p, s)
            if self.show_thoughts:
                for side in s.pilots:
                    self.paint_thoughts(p, s, side)
            self.paint_score(p, s, pal)
        if self.banner:
            f = QFont(self.font())
            f.setPointSize(22)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(pal["text"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.banner)
        p.end()

    def c(self, key):
        return QColor(COLORS.get(self.theme, COLORS["dark"])[key])

    def circle(self, p, x, y, r, fill, edge=None, width=2.0):
        pt = self.to_screen(x, y)
        rr = r * self.scale()
        p.setBrush(QBrush(fill))
        p.setPen(QPen(edge, width) if edge is not None else Qt.PenStyle.NoPen)
        p.drawEllipse(pt, rr, rr)

    def paint_hockey(self, p, s):
        w = s.world
        hp = w.p
        k = self.scale()
        tl, br = self.to_screen(0, 0), self.to_screen(hp.X, hp.Y)
        rink = QRectF(tl, br)
        p.setPen(QPen(self.c("rink_line"), 3))
        p.setBrush(self.c("rink"))
        p.drawRoundedRect(rink, 0.08 * k, 0.08 * k)
        p.setPen(QPen(self.c("center"), 2))
        p.drawLine(self.to_screen(hp.X / 2, 0), self.to_screen(hp.X / 2, hp.Y))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(self.to_screen(hp.X / 2, hp.Y / 2), 0.15 * k, 0.15 * k)
        p.setPen(QPen(self.c("rink_line"), 1, Qt.PenStyle.DashLine))
        for x in (hp.X / 4, 3 * hp.X / 4):
            p.drawLine(self.to_screen(x, 0), self.to_screen(x, hp.Y))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.c("goal"))
        for x0 in (-0.05, hp.X):
            p.drawRect(QRectF(self.to_screen(x0, hp.goal_lo), self.to_screen(x0 + 0.05, hp.goal_hi)))
        for side in (0, 1):
            mx, my = w.mal[0, side]
            col = self.c("left" if side == 0 else "right")
            self.circle(p, mx, my, hp.R, col, col.darker(140), 2)
            self.circle(p, mx, my, hp.R * 0.45, col.lighter(130), col.darker(160), 1.5)
        px, py = w.puck[0]
        self.circle(p, px, py, hp.r, self.c("puck"), self.c("puck_edge"), 1.5)

    def paint_pong(self, p, s):
        w = s.world
        pp = w.p
        k = self.scale()
        # пол
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.c("floor"))
        p.drawRect(QRectF(self.to_screen(PONG_VIEW[0], pp.floor + 0.02), self.to_screen(PONG_VIEW[1], pp.floor)))
        # стол: столешница и ножки
        p.setBrush(self.c("table"))
        p.drawRect(QRectF(self.to_screen(0, 0), self.to_screen(pp.L, -0.035)))
        p.setBrush(self.c("table").darker(150))
        for x in (0.25, pp.L - 0.29):
            p.drawRect(QRectF(self.to_screen(x, -0.035), self.to_screen(x + 0.04, pp.floor)))
        p.setPen(QPen(self.c("table_edge"), 2))
        p.drawLine(self.to_screen(0, 0), self.to_screen(pp.L, 0))
        # сетка
        p.setPen(QPen(self.c("net"), max(2.0, 0.012 * k)))
        p.drawLine(self.to_screen(pp.L / 2, 0), self.to_screen(pp.L / 2, pp.net_h))
        # ракетки
        for side in (0, 1):
            cx, cy = w.pc[0, side]
            th = w.th[0, side]
            n = pong.normal(th, side)
            ux, uy = -n[1] * pp.half, n[0] * pp.half
            col = self.c("left" if side == 0 else "right")
            p.setPen(QPen(col, max(4.0, 0.03 * k), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(self.to_screen(cx - ux, cy - uy), self.to_screen(cx + ux, cy + uy))
            hx, hy = cx - n[0] * 0.02 - ux * 1.0, cy - n[1] * 0.02 - uy * 1.0
            p.setPen(QPen(self.c("handle"), max(3.0, 0.02 * k), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(self.to_screen(cx - ux, cy - uy), self.to_screen(hx - ux * 0.6, hy - uy * 0.6))
        bx, by = w.ball[0]
        self.circle(p, bx, by, pp.r * 1.6, self.c("ball"), self.c("ball").darker(150), 1)

    def paint_thoughts(self, p, s, side):
        got = s.thoughts(side)
        if not got:
            return
        path, aim = got
        col = self.c("path")
        if len(path) > 1:
            pen = QPen(col, 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            pp = QPainterPath(self.to_screen(*path[0]))
            for x, y in path[1:]:
                pp.lineTo(self.to_screen(x, y))
            p.drawPath(pp)
        if aim is not None:
            p.setPen(QPen(self.c("aim"), 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self.to_screen(*aim), 9, 9)

    def paint_score(self, p, s, pal):
        a, b = s.world.score[0]
        f = QFont(self.font())
        f.setPointSize(20)
        f.setBold(True)
        p.setFont(f)
        rect, *_ = self.frame()
        p.setPen(QColor(pal["text"]))
        box = QRectF(rect.left(), rect.top() + 4, rect.width(), 40)
        p.drawText(box, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, f"{a} : {b}")
