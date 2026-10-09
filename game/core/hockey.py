"""Аэрохоккей, вид сверху. Своя 2D-физика с фиксированным шагом 1/240 с, n партий сразу.

Поле - X × Y (2 × 1), ворота в коротких бортах. Сторона 0 - слева, сторона 1 - справа.
Бита - круг, которым управляют ускорением: команда (n, 2, 2) - для каждой партии и каждой
стороны вектор не длиннее 1, в долях A_MAX. Скорость биты - не больше V_MAX. Эти пределы
одни для человека и для ИИ: мышь тоже превращается в команду ускорения (game/core/control.py).

Шайба скользит с малым трением, отскакивает от бортов и бит. Удары ищутся непрерывно
(game/core/geom.py): за шаг шайба может пройти больше своего размера, но сквозь борт или
биту не пролетит. Гол - центр шайбы перешёл линию ворот. Шайба, простоявшая на одной
половине 7 с, переходит сопернику - иначе партия «ИИ против ИИ» могла бы встать.
"""
from dataclasses import dataclass

import numpy as np

from game.core.geom import clip_norm, contact, norm, toi_point, toi_segment

TICK = 1.0 / 240.0


@dataclass(frozen=True)
class Params:
    X: float = 2.0                  # длина поля
    Y: float = 1.0                  # ширина
    goal_lo: float = 0.32           # ворота: по y от goal_lo до goal_hi
    goal_hi: float = 0.68
    r: float = 0.032                # шайба
    R: float = 0.05                 # бита
    damping: float = 0.08           # трение шайбы, 1/с: скорость * exp(-damping * t)
    e_wall: float = 0.92
    e_mallet: float = 0.9
    puck_max: float = 8.0
    v_max: float = 3.5              # бита: скорость
    a_max: float = 40.0             # бита: ускорение
    stall: float = 7.0              # секунд на одной половине
    to_win: int = 7
    substeps: int = 4               # ударов за один шаг, больше - редкость (зажатая шайба)


P = Params()


def walls(p=P):
    """Отрезки бортов (k, 2, 2) и открытые концы бортов у ворот - штанги (4, 2)."""
    X, Y, g0, g1 = p.X, p.Y, p.goal_lo, p.goal_hi
    segs = np.array([
        [[-1.0, 0.0], [X + 1.0, 0.0]],
        [[-1.0, Y], [X + 1.0, Y]],
        [[0.0, 0.0], [0.0, g0]], [[0.0, g1], [0.0, Y]],
        [[X, 0.0], [X, g0]], [[X, g1], [X, Y]],
    ])
    posts = np.array([[0.0, g0], [0.0, g1], [X, g0], [X, g1]])
    return segs, posts


def mallet_box(p=P):
    """Где может стоять центр биты: (2 стороны, [xmin, xmax, ymin, ymax])."""
    return np.array([[p.R, p.X / 2 - p.R, p.R, p.Y - p.R],
                     [p.X / 2 + p.R, p.X - p.R, p.R, p.Y - p.R]])


def home(p=P):
    return np.array([[0.16 * p.X / 2, p.Y / 2], [p.X - 0.16 * p.X / 2, p.Y / 2]])


def spawn(side, p=P):
    """Где кладётся шайба стороне side (после гола в её ворота или по правилу 7 секунд)."""
    return np.array([p.X / 4, p.Y / 2]) if side == 0 else np.array([3 * p.X / 4, p.Y / 2])


class Hockey:
    """n партий аэрохоккея. Состояние - массивы numpy; step(cmd) - один шаг 1/240 с."""

    game = "hockey"

    def __init__(self, n=1, params=P, first=None, seed=0):
        self.p = params
        self.n = n
        self.segs, self.posts = walls(params)
        self.box = mallet_box(params)
        rng = np.random.default_rng(seed)
        self.mal = np.repeat(home(params)[None], n, axis=0)
        self.mv = np.zeros((n, 2, 2))
        self.puck = np.zeros((n, 2))
        self.pv = np.zeros((n, 2))
        self.score = np.zeros((n, 2), dtype=np.int64)
        self.over = np.zeros(n, dtype=bool)
        self.winner = np.full(n, -1)
        self.ticks = 0
        self.stall_t = np.zeros(n)
        self.half = np.zeros(n, dtype=np.int64)
        first = rng.integers(0, 2, n) if first is None else np.broadcast_to(np.asarray(first), (n,))
        for side in (0, 1):
            m = first == side
            self.puck[m] = spawn(side, params)
            self.half[m] = side
        self.hit = np.zeros((n, 2), dtype=bool)
        self.wall = np.zeros(n, dtype=bool)
        self.goal = np.full(n, -1)

    # --- биты ---

    def move_mallets(self, cmd):
        """Ускорение не больше A_MAX, скорость не больше V_MAX, бита в своей половине.
        Возвращает (где были, где стали, средняя скорость за шаг)."""
        p = self.p
        acc = clip_norm(np.asarray(cmd, dtype=float), 1.0) * p.a_max
        v = clip_norm(self.mv + acc * TICK, p.v_max)
        m0 = self.mal
        m1 = m0 + v * TICK
        lo = self.box[None, :, [0, 2]]
        hi = self.box[None, :, [1, 3]]
        clamped = np.clip(m1, lo, hi)
        v = np.where(clamped != m1, 0.0, v)
        self.mv = v
        self.mal = clamped
        return m0, clamped, (clamped - m0) / TICK

    # --- шаг ---

    def step(self, cmd):
        p = self.p
        live = ~self.over
        cmd = np.where(live[:, None, None], cmd, 0.0)
        m0, m1, vm = self.move_mallets(cmd)
        self.hit = np.zeros((self.n, 2), dtype=bool)
        self.wall = np.zeros(self.n, dtype=bool)
        self.goal = np.full(self.n, -1)

        pos = self.puck.copy()
        vel = self.pv * np.exp(-p.damping * TICK)
        vel = np.where(live[:, None], vel, 0.0)
        tau = np.where(live, TICK, 0.0)
        s = np.zeros(self.n)                          # прошедшее время внутри шага
        for _ in range(p.substeps):
            if not (tau > 0).any():
                break
            # борта: (n, 6)
            tw, nw = toi_segment(pos[:, None, :], vel[:, None, :], self.segs[None, :, 0], self.segs[None, :, 1],
                                 p.r, tau[:, None])
            # штанги: (n, 4)
            dp = pos[:, None, :] - self.posts[None]
            tp = toi_point(dp, vel[:, None, :], p.r, tau[:, None])
            # биты: (n, 2), положение биты в момент s внутри шага
            mk = m0 + vm * s[:, None, None]
            dm = pos[:, None, :] - mk
            wm = vel[:, None, :] - vm
            tm = toi_point(dm, wm, p.r + p.R, tau[:, None])
            allt = np.concatenate([tw, tp, tm], axis=1)
            which = allt.argmin(axis=1)
            tmin = allt[np.arange(self.n), which]
            hit = np.isfinite(tmin) & (tau > 0)
            adv = np.where(hit, tmin, tau)
            pos = pos + vel * adv[:, None]
            s = s + adv
            tau = np.where(hit, tau - adv, 0.0)
            if not hit.any():
                break
            idx = np.nonzero(hit)[0]
            w = which[idx]
            nrm = np.zeros((len(idx), 2))
            surf = np.zeros((len(idx), 2))
            e = np.full(len(idx), p.e_wall)
            kw = w < 6
            nrm[kw] = nw[idx[kw], w[kw]]
            kp = (w >= 6) & (w < 10)
            if kp.any():
                d = pos[idx[kp]] - self.posts[w[kp] - 6]
                nrm[kp] = d / norm(d)[:, None]
            km = w >= 10
            if km.any():
                side = w[km] - 10
                mpos = m0[idx[km], side] + vm[idx[km], side] * s[idx[km], None]
                d = pos[idx[km]] - mpos
                nrm[km] = d / np.maximum(norm(d), 1e-12)[:, None]
                surf[km] = vm[idx[km], side]
                e[km] = p.e_mallet
                self.hit[idx[km], side] = True
            self.wall[idx[~km]] = True
            v_new, _ = contact(vel[idx], np.zeros(len(idx)), nrm, surf, e, 0.0, 2.0 / 3.0, p.r)
            vel[idx] = v_new
        # Страховка на редкий случай, когда ударов за шаг больше substeps (шайба зажата
        # битой у борта): вытолкнуть из бит и вернуть в поле. Скорость не трогается.
        for side in (0, 1):
            d = pos - m1[:, side]
            dist = norm(d)
            inside = live & (dist < p.r + p.R)
            if inside.any():
                safe = np.where(dist > 1e-9, dist, 1.0)
                dirn = np.where((dist > 1e-9)[:, None], d / safe[:, None], np.array([1.0 - 2 * side, 0.0]))
                pos = np.where(inside[:, None], m1[:, side] + dirn * (p.r + p.R), pos)
        in_mouth = (pos[:, 1] > p.goal_lo) & (pos[:, 1] < p.goal_hi)
        pos[:, 1] = np.clip(pos[:, 1], p.r, p.Y - p.r)
        pos[:, 0] = np.where(in_mouth, pos[:, 0], np.clip(pos[:, 0], p.r, p.X - p.r))
        vel = clip_norm(vel, p.puck_max)
        self.puck = np.where(live[:, None], pos, self.puck)
        self.pv = np.where(live[:, None], vel, self.pv)
        self.ticks += 1
        self._rules(live)

    def _rules(self, live):
        p = self.p
        x = self.puck[:, 0]
        scorer = np.where(x < 0, 1, np.where(x > p.X, 0, -1))
        scorer = np.where(live, scorer, -1)
        self.goal = scorer
        half = (x >= p.X / 2).astype(np.int64)
        self.stall_t = np.where(half == self.half, self.stall_t + TICK, 0.0)
        self.half = half
        for side in (0, 1):
            m = scorer == side
            if m.any():
                self.score[m, side] += 1
                self._serve(m, 1 - side)
        stalled = live & (scorer < 0) & (self.stall_t >= p.stall)
        for side in (0, 1):
            m = stalled & (self.half == side)
            if m.any():
                self._serve(m, 1 - side, reset_mallets=False)
        won = live & (self.score.max(axis=1) >= p.to_win)
        if won.any():
            self.over |= won
            self.winner = np.where(won, self.score.argmax(axis=1), self.winner)

    def _serve(self, mask, side, reset_mallets=True):
        """Шайба - стороне side, на её половину, неподвижно."""
        self.puck[mask] = spawn(side, self.p)
        self.pv[mask] = 0.0
        self.half[mask] = side
        self.stall_t[mask] = 0.0
        if reset_mallets:
            self.mal[mask] = home(self.p)
            self.mv[mask] = 0.0

    # --- для окна, проверок и ИИ ---

    def snapshot(self, i=0):
        """Состояние одной партии - словарем, для окна и сравнений."""
        return {"puck": self.puck[i].copy(), "pv": self.pv[i].copy(), "mal": self.mal[i].copy(),
                "mv": self.mv[i].copy(), "score": self.score[i].copy(), "over": bool(self.over[i])}

    def place(self, i, puck=None, pv=None, mal=None, mv=None):
        """Поставить шайбу и биты руками - для проверок."""
        if puck is not None:
            self.puck[i] = puck
            self.half[i] = int(self.puck[i, 0] >= self.p.X / 2)
            self.stall_t[i] = 0.0
        if pv is not None:
            self.pv[i] = pv
        if mal is not None:
            self.mal[i] = mal
        if mv is not None:
            self.mv[i] = mv

