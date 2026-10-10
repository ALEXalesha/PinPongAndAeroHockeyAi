"""Явный прогноз траектории - подсказка сети и основа бота-«предсказателя».

Всё в зеркальной системе игрока: свои ворота или свой край стола - у x = 0.

Аэрохоккей: шайба летит по прямой с экспоненциальным трением, от длинных бортов
отражается. Где она пересечёт линию защиты x = xd и когда - считается формулой: путь по x и
по y при трении уменьшается в одной пропорции, поэтому y на линии защиты - это y + vy·dx/|vx|,
сложенный «гармошкой» между бортами (каждое отражение - зеркало).

Пинг-понг: мяч - по дуге (тяжесть), сопротивление воздуха приближено экспонентой по x,
вращения ИИ не видит и не учитывает - поэтому на крученых мячах прогноз ошибается, как
ошибается и человек. Ищется точка, где мяч после отскока на своей стороне пересечёт линию
удара x = x_hit за краем стола; если до неё мяч не долетает высоко - верхняя точка отскока.
"""
import numpy as np

from game.core import hockey, pong


def fold(v, lo, hi):
    """Отражения между стенками lo и hi: «гармошка»."""
    span = hi - lo
    m = np.mod(v - lo, 2 * span)
    return np.where(m > span, 2 * span - m, m) + lo


# --- аэрохоккей ---

HOCKEY_XD = 0.16


def hockey_intercept(x, y, vx, vy, xd=HOCKEY_XD, p=hockey.P):
    """(y на линии защиты, время, долетит ли). Шайба, идущая от ворот или не долетающая
    из-за трения, - не долетит."""
    k = p.damping
    dx = x - xd
    coming = (vx < -1e-6) & (dx > 0)
    speed = np.where(coming, -vx, 1.0)
    ratio = k * dx / speed
    reach = coming & (ratio < 0.999)
    t = np.where(reach, -np.log1p(-np.minimum(ratio, 0.999)) / k, np.inf)
    yy = fold(y + vy * dx / speed, p.r, p.Y - p.r)
    return np.where(reach, yy, y), t, reach


def hockey_path(x, y, vx, vy, xd=HOCKEY_XD, p=hockey.P, points=48):
    """Ломаная пути шайбы до линии защиты - для «Мыслей ИИ» (одна партия, числа)."""
    if vx >= -1e-6 or x <= xd:
        return []
    k = p.damping
    dx = x - xd
    if k * dx / -vx >= 0.999:
        dx = 0.999 * -vx / k
    s = np.linspace(0, 1, points)
    xs = x - dx * s
    ys = fold(y + vy * dx / -vx * s, p.r, p.Y - p.r)
    return list(zip(xs.tolist(), ys.tolist()))


# --- пинг-понг ---

PONG_HIT_X = -0.2
PONG_HOME = (-0.35, 0.22)


def _ballistic_x(x0, vx0, c, t):
    """x(t) при торможении vx' = -c vx."""
    return x0 + vx0 * np.where(c > 1e-9, -np.expm1(-c * t) / np.maximum(c, 1e-9), t)


def _time_to_x(x0, vx0, c, x1):
    """Когда x станет x1 (vx0 к x1); inf - не долетит."""
    dx = x1 - x0
    ok = (dx * vx0 > 0)
    ratio = np.where(ok, c * dx / np.where(ok, vx0, 1.0), 2.0)
    t = np.where(ok & (ratio < 0.999), -np.log1p(-np.minimum(ratio, 0.999)) / np.maximum(c, 1e-9), np.inf)
    t = np.where(ok & (c <= 1e-9), dx / np.where(ok, vx0, 1.0), t)
    return t


def pong_intercept(x, y, vx, vy, bounced_mine, p=pong.P, x_hit=PONG_HIT_X):
    """Где и когда бить: (hx, hy, t, valid). Все массивы в зеркальной системе игрока.

    bounced_mine - мяч уже отскочил на моей стороне после удара соперника."""
    g, r = p.g, p.r
    speed = np.sqrt(vx * vx + vy * vy)
    c = p.drag * speed
    coming = vx < -0.05
    # 1. отскок на моей стороне, если его ещё не было
    t1 = (vy + np.sqrt(np.maximum(vy * vy + 2 * g * (y - r), 0.0))) / g
    x1 = _ballistic_x(x, vx, c, t1)
    on_mine = (x1 >= 0) & (x1 <= p.L / 2)
    vx_hit = vx * np.exp(-c * t1)
    vy_in = vy - g * t1
    vy1 = -vy_in * p.e_table
    dvx = np.minimum(np.abs(vx_hit) / (1 + 1 / p.k), p.mu_table * (1 + p.e_table) * np.abs(vy_in))
    vx1 = vx_hit - np.sign(vx_hit) * dvx
    # вращение, которое мяч получит от стола (до отскока считаем его нулём - не видно),
    # и его Магнус после отскока: постоянная добавка к тяжести
    spin1 = -np.sign(vx_hit) * dvx / (p.k * p.r)
    # начальная точка второго отрезка: после отскока или прямо сейчас (уже отскочил)
    bx = np.where(bounced_mine, x, x1)
    by = np.where(bounced_mine, y, r)
    bvx = np.where(bounced_mine, vx, vx1)
    bvy = np.where(bounced_mine, vy, vy1)
    t0 = np.where(bounced_mine, 0.0, t1)
    c2 = p.drag * np.sqrt(bvx * bvx + bvy * bvy)
    g = g - np.where(bounced_mine, 0.0, p.magnus * spin1 * bvx)
    # 2. пересечение линии удара
    tau = _time_to_x(bx, bvx, c2, x_hit)
    fin = np.isfinite(tau)
    tau = np.where(fin, tau, 0.0)
    hy = by + bvy * tau - 0.5 * g * tau * tau
    cross = fin & (hy > r + 0.02)
    # не долетает высоко - верхняя точка после отскока
    ta = np.maximum(bvy / g, 0.0)
    ax = _ballistic_x(bx, bvx, c2, ta)
    ay = by + bvy * ta - 0.5 * g * ta * ta
    hx = np.where(cross, x_hit, ax)
    hy = np.where(cross, hy, ay)
    t = t0 + np.where(cross, tau, ta)
    valid = coming & (bounced_mine | on_mine) & (hx < 0.3)
    hx = np.where(valid, hx, PONG_HOME[0])
    hy = np.where(valid, hy, PONG_HOME[1])
    t = np.where(valid, t, 0.0)
    return hx, hy, t, valid


def pong_path(x, y, vx, vy, bounced_mine, p=pong.P, horizon=1.0, points=60):
    """Точки дуги мяча для «Мыслей ИИ»: тот же упрощённый расчёт (без вращения), шаг за шагом."""
    out = []
    dt = horizon / points
    bounced = bool(bounced_mine)
    for _ in range(points):
        speed = (vx * vx + vy * vy) ** 0.5
        vx -= p.drag * speed * vx * dt
        vy -= p.g * dt
        x += vx * dt
        y += vy * dt
        if y < p.r and 0 <= x <= p.L and vy < 0:
            if bounced or x > p.L / 2:
                out.append((x, p.r))
                break
            y = p.r
            vy_in = vy
            vy = -vy * p.e_table
            dvx = min(abs(vx) / (1 + 1 / p.k), p.mu_table * (1 + p.e_table) * abs(vy_in))
            vx -= np.sign(vx) * dvx
            bounced = True
        out.append((x, y))
        if y < p.floor or x < -1.2:
            break
    return out
