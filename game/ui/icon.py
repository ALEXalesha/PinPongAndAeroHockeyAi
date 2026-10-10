"""Иконка: ракетка и мяч на синем столе. Её же собирает в .ico build.py."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap


def render(size=256):
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0, 0, size, size), size * 0.18, size * 0.18)
    p.setClipPath(clip)
    p.fillRect(QRectF(0, 0, size, size), QColor("#1d4f86"))
    p.setPen(QPen(QColor("#e9eef5"), size * 0.025))
    p.drawLine(QPointF(0, size * 0.72), QPointF(size, size * 0.72))
    p.drawLine(QPointF(size * 0.5, size * 0.72), QPointF(size * 0.5, size * 0.58))
    # ракетка
    p.setPen(QPen(QColor("#c8a27a"), size * 0.07, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(size * 0.33, size * 0.62), QPointF(size * 0.2, size * 0.78))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#e0525f"))
    p.drawEllipse(QPointF(size * 0.4, size * 0.47), size * 0.17, size * 0.19)
    # мяч
    p.setBrush(QColor("#ff9f1c"))
    p.drawEllipse(QPointF(size * 0.7, size * 0.3), size * 0.08, size * 0.08)
    p.end()
    return img


def app_icon():
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(render(s)))
    return icon
