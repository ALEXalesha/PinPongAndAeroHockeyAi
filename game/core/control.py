"""Управление и часы игры.

seek - как мышь превращается в команду ускорения. Человек показывает мышью точку, бита
туда едет, но с теми же пределами скорости и ускорения, что у ИИ: разгон и торможение по
профилю «успеть остановиться». Так честность ИИ симметрична: у человека нет биты, которая
телепортируется за курсором.

Clock - фиксированный шаг физики независимо от частоты кадров. Окно сообщает, сколько
прошло настоящего времени; часы отдают целое число шагов по 1/240 с, остаток копится.
Сколько бы кадров ни было за секунду - 30 или 144, - за секунду будет 240 шагов, и при тех
же командах партия пройдёт одинаково (закон в tests/test_control.py).
"""
import numpy as np

from game.core.geom import clip_norm, norm


def seek(pos, vel, target, a_max, v_max, tick, gain=0.9):
    """Команда ускорения (доли a_max, вектор не длиннее 1), чтобы приехать в target и
    остановиться. Работает с любыми ведущими осями: pos, vel, target - (..., 2)."""
    d = target - pos
    dist = norm(d)
    # скорость, с которой ещё можно затормозить до точки: v² = 2 a s
    speed = np.minimum(v_max, np.sqrt(2.0 * a_max * gain * dist))
    want = d / np.maximum(dist, 1e-9)[..., None] * speed[..., None]
    acc = (want - vel) / tick
    return clip_norm(acc / a_max, 1.0)


def seek1(x, v, target, a_max, v_max, tick, gain=0.9):
    """То же по одной оси (наклон ракетки): команда в [-1, 1]."""
    d = target - x
    speed = np.minimum(v_max, np.sqrt(2.0 * a_max * gain * np.abs(d)))
    want = np.sign(d) * speed
    return np.clip((want - v) / tick / a_max, -1.0, 1.0)


class Clock:
    """Накопитель времени: feed(секунды) -> сколько шагов сделать сейчас.

    max_steps ограничивает догон после долгой паузы (окно свернули, отладчик): иначе
    игра пыталась бы прожить разом минуту и замерла бы."""

    def __init__(self, tick, max_steps=24):
        self.tick = tick
        self.max_steps = max_steps
        self.total = 0.0          # сколько времени прожито по часам окна
        self.steps = 0            # сколько шагов отдано

    def feed(self, seconds):
        """Шаги считаются от общего времени, а не от остатка: так число шагов за одно и то
        же время не зависит от того, какими кусками оно пришло."""
        self.total += max(0.0, float(seconds))
        k = int(self.total / self.tick + 1e-6) - self.steps
        if k > self.max_steps:
            k = self.max_steps
            self.total = (self.steps + k) * self.tick
        self.steps += k
        return k
