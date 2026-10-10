"""Игроки за столом: боты и сеть, задержка реакции, шум, частота решений.

Pilot ведёт одну сторону в части партий (rows). Каждые period шагов он «смотрит»
(Eyes.look с задержкой delay и шумом noise) и решает. Решение бывает двух видов:
- сеть (Brain) выдаёт ускорение - оно держится до следующего решения;
- бот выдаёт точку и наклон - туда ракетку каждый шаг ведёт seek, как мышь человека.

В обоих случаях ракетка двигается через ту же физику с теми же пределами, что у человека.
"""
import numpy as np

from game.ai import predict
from game.ai.observe import Eyes, hockey_row, pong_row
from game.core import hockey, pong
from game.core.control import seek, seek1

HOCKEY_F = 16
PONG_F = 20
ACTIONS = {"hockey": 2, "pong": 3}
FEATURES = {"hockey": HOCKEY_F, "pong": PONG_F}
HINT = {"hockey": slice(12, 16), "pong": slice(14, 18)}


def make_eyes(world):
    row = hockey_row(world) if world.game == "hockey" else pong_row(world)
    eyes = Eyes(world.n, row.shape[1])
    eyes.reset(row)
    return eyes


def see(world, eyes):
    eyes.push(hockey_row(world) if world.game == "hockey" else pong_row(world))


# --- что видит сторона: всё в её зеркальной системе ---


def hash_normal(ids, count, k):
    """Нормальный шум (len(ids), k), который зависит только от номера партии и номера решения:
    партия i в пачке из 200 и та же партия одна получают один и тот же шум. Хэш - как в
    шейдерах (дробная часть синуса), нормальность - по Боксу-Мюллеру."""
    j = np.arange(2 * k)[None, :]
    x = ids[:, None].astype(np.float64) * 12.9898 + (count * 2 * k + j) * 78.233
    u = np.mod(np.sin(np.mod(x, 6283.185307179586)) * 43758.5453, 1.0)
    u1 = np.clip(u[:, :k], 1e-12, 1.0)
    u2 = u[:, k:]
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


def hockey_view(w, view, rows, side, eps):
    p = w.p
    sx = 1.0 if side == 0 else -1.0
    px = view[:, 0] if side == 0 else p.X - view[:, 0]
    py = view[:, 1].copy()
    pvx, pvy = view[:, 2] * sx, view[:, 3].copy()
    if eps is not None:
        e = eps
        px, py = px + e[:, 0], py + e[:, 1]
        pvx, pvy = pvx + 8 * e[:, 2], pvy + 8 * e[:, 3]
    o = 1 - side
    ox = view[:, 4 + 2 * o] if side == 0 else p.X - view[:, 4 + 2 * o]
    oy = view[:, 5 + 2 * o]
    mx = w.mal[rows, side, 0] if side == 0 else p.X - w.mal[rows, side, 0]
    my = w.mal[rows, side, 1]
    mvx, mvy = w.mv[rows, side, 0] * sx, w.mv[rows, side, 1]
    hy, ht, valid = predict.hockey_intercept(px, py, pvx, pvy, p=p)
    hx = np.where(valid, predict.HOCKEY_XD, px)
    return dict(px=px, py=py, pvx=pvx, pvy=pvy, mx=mx, my=my, mvx=mvx, mvy=mvy, ox=ox, oy=oy,
                hx=hx, hy=hy, ht=np.where(valid, ht, 0.0), valid=valid)


def hockey_vec(v, hint=True):
    p = hockey.P
    f = np.stack([v["px"] / p.X, v["py"] / p.Y, v["pvx"] / 8, v["pvy"] / 8,
                  v["mx"] / p.X, v["my"] / p.Y, v["mvx"] / p.v_max, v["mvy"] / p.v_max,
                  v["ox"] / p.X, v["oy"] / p.Y, (v["px"] - v["mx"]) * 2, (v["py"] - v["my"]) * 2,
                  (v["hx"] - v["mx"]) * 2, (v["hy"] - v["my"]) * 2, np.minimum(v["ht"], 2.0) / 2,
                  v["valid"].astype(float)], axis=1)
    if not hint:
        f[:, HINT["hockey"]] = 0.0
    return f


def pong_view(w, view, rows, side, eps):
    p = w.p
    sx = 1.0 if side == 0 else -1.0
    bx = view[:, 0] if side == 0 else p.L - view[:, 0]
    by = view[:, 1].copy()
    bvx, bvy = view[:, 2] * sx, view[:, 3].copy()
    if eps is not None:
        e = eps
        bx, by = bx + e[:, 0], by + e[:, 1]
        bvx, bvy = bvx + 6 * e[:, 2], bvy + 6 * e[:, 3]
    o = 1 - side
    oy = view[:, 5 + 2 * o]
    phase, server, last, bounced = view[:, 10], view[:, 11], view[:, 12], view[:, 13]
    px = w.pc[rows, side, 0] if side == 0 else p.L - w.pc[rows, side, 0]
    py = w.pc[rows, side, 1]
    pvx, pvy = w.pv[rows, side, 0] * sx, w.pv[rows, side, 1]
    th, tw = w.th[rows, side], w.tw[rows, side]
    mine = server == side
    serving = mine & ((phase == pong.SERVE) | ((phase == pong.PLAY) & (last == -1)))
    bounced_mine = (bounced == side) & (last == o)
    hx, hy, ht, valid = predict.pong_intercept(bx, by, bvx, bvy, bounced_mine, p=p)
    # своя подача: бить по подброшенному мячу, когда он опустится к высоте подброса
    sy = p.toss_y - 0.05
    resting = phase == pong.SERVE
    fall = (bvy + np.sqrt(np.maximum(bvy * bvy + 2 * p.g * (by - sy), 0.0))) / p.g
    hx = np.where(serving, bx, hx)
    hy = np.where(serving, sy, hy)
    ht = np.where(serving, np.where(resting, 1.0, fall), ht)
    valid = valid | serving
    return dict(bx=bx, by=by, bvx=bvx, bvy=bvy, px=px, py=py, pvx=pvx, pvy=pvy, th=th, tw=tw, oy=oy,
                hx=hx, hy=hy, ht=ht, valid=valid, serving=serving, bounced_mine=bounced_mine)


def pong_vec(v, hint=True):
    p = pong.P
    f = np.stack([v["bx"] / 1.4, v["by"], v["bvx"] / 8, v["bvy"] / 8,
                  v["px"], v["py"], v["pvx"] / p.v_max, v["pvy"] / p.v_max, v["th"] / p.tilt_max,
                  v["tw"] / p.w_max, v["oy"], (v["bx"] - v["px"]) * 2, (v["by"] - v["py"]) * 2,
                  (v["bvx"] < 0).astype(float),
                  (v["hx"] - v["px"]) * 2, (v["hy"] - v["py"]) * 2, np.minimum(v["ht"], 1.5) / 1.5,
                  v["valid"].astype(float), v["serving"].astype(float), v["bounced_mine"].astype(float)],
                 axis=1)
    if not hint:
        f[:, HINT["pong"]] = 0.0
    return f


# --- боты ---


class Follower:
    """«Следящий»: бита на линии защиты ходит за шайбой по y; шайба на своей половине и
    медленная - бьёт по ней. В пинг-понге - ракетка по высоте мяча за краем стола."""

    kind = "target"

    def decide(self, game, v):
        if game == "hockey":
            slow = (np.hypot(v["pvx"], v["pvy"]) < 1.0) & (v["px"] < hockey.P.X / 2)
            tx = np.where(slow, v["px"] + 0.05, predict.HOCKEY_XD)
            ty = v["py"]
            return tx, ty, None
        near = v["bx"] < 0.25
        tx = np.where(v["serving"] | near, v["bx"] + 0.1, -0.35)
        ty = np.clip(v["by"], 0.05, 0.6)
        return tx, ty, np.full_like(tx, 0.25)


class Predictor:
    """«Предсказатель»: идёт в точку, где шайба пересечёт линию защиты (где мяч будет
    на линии удара), и бьёт оттуда. В аэрохоккее медленную шайбу добивает к чужим воротам."""

    kind = "target"

    # подобрано перебором (tools в истории сессии): подача 100% на сторону соперника,
    # в розыгрыше 1.7 удачных удара на очко у двух таких ботов
    def __init__(self, swing=0.12, lead=0.1, tilt=(0.42, 0.12, -0.08), serve_tilt=0.45):
        self.swing, self.lead, self.tilt, self.serve_tilt = swing, lead, tilt, serve_tilt

    def decide(self, game, v):
        if game == "hockey":
            p = hockey.P
            slow = (np.hypot(v["pvx"], v["pvy"]) < 1.2) & (v["px"] < p.X / 2)
            # сзади шайбы на линии к дальнему от вратаря углу ворот, потом - сквозь неё
            gx = p.X
            gy = np.where(v["oy"] > p.Y / 2, p.goal_lo + 0.06, p.goal_hi - 0.06)
            d = np.stack([gx - v["px"], gy - v["py"]], axis=1)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            behind = v["mx"] < v["px"] - 0.02
            back = p.r + p.R + 0.04
            sx = np.where(behind, v["px"] + d[:, 0] * 0.1, v["px"] - d[:, 0] * back)
            sy = np.where(behind, v["py"] + d[:, 1] * 0.1, v["py"] - d[:, 1] * back)
            tx = np.where(slow, sx, np.where(v["valid"], v["hx"], predict.HOCKEY_XD))
            ty = np.where(slow, sy, np.where(v["valid"], v["hy"], v["py"]))
            return tx, ty, None
        # пинг-понг: позади точки удара, за lead секунд до удара - замах вперёд
        late = v["ht"] < self.lead
        tx = np.where(late, v["hx"] + self.swing, v["hx"] - self.swing)
        tx = np.where(v["valid"], tx, predict.PONG_HOME[0])
        ty = np.where(v["valid"], v["hy"], predict.PONG_HOME[1])
        low, mid, high = self.tilt
        tilt = np.where(v["hy"] < 0.12, low, np.where(v["hy"] < 0.35, mid, high))
        tilt = np.where(v["serving"], self.serve_tilt, tilt)
        return tx, ty, tilt


class Idle:
    kind = "target"

    def decide(self, game, v):
        if game == "hockey":
            return np.full_like(v["px"], predict.HOCKEY_XD), np.full_like(v["px"], hockey.P.Y / 2), None
        return (np.full_like(v["bx"], predict.PONG_HOME[0]), np.full_like(v["bx"], predict.PONG_HOME[1]),
                np.zeros_like(v["bx"]))


# --- сеть ---


def sizes(game, hidden=(16, 16)):
    return (FEATURES[game],) + tuple(hidden) + (ACTIONS[game],)


def param_count(shape):
    return sum(a * b + b for a, b in zip(shape[:-1], shape[1:]))


class Brain:
    """MLP на numpy: tanh в скрытых слоях и на выходе. Веса - плоские векторы, по одному на
    члена популяции: (M, D). Партия i играет весами member[i]."""

    kind = "acc"

    def __init__(self, game, params, hidden=(16, 16), hint=True):
        self.game = game
        self.shape = sizes(game, hidden)
        self.params = np.atleast_2d(np.asarray(params, dtype=float))
        self.hint = hint
        assert self.params.shape[1] == param_count(self.shape)
        self.layers = []
        k = 0
        for a, b in zip(self.shape[:-1], self.shape[1:]):
            W = self.params[:, k:k + a * b].reshape(-1, a, b)
            k += a * b
            bias = self.params[:, k:k + b]
            k += b
            self.layers.append((W, bias))

    def forward(self, x, member):
        h = x
        for W, b in self.layers:
            if len(W) == 1:
                h = np.tanh(h @ W[0] + b[0])
            else:
                h = np.tanh(np.einsum("nf,nfh->nh", h, W[member]) + b[member])
        return h


# --- пилот одной стороны ---


class Pilot:
    def __init__(self, world, side, policy, rows=None, delay=24, noise=0.0, period=4, seed=0, member=None,
                 hint=True, ids=None):
        self.w = world
        self.game = world.game
        self.side = side
        self.policy = policy
        self.rows = np.arange(world.n) if rows is None else np.asarray(rows)
        m = len(self.rows)
        self.delay = np.broadcast_to(np.asarray(delay), (m,)).astype(np.int64)
        self.noise = np.broadcast_to(np.asarray(noise, dtype=float), (m,))
        self.period = period
        # номера партий для шума: по умолчанию номер строки и зерно стороны
        base = np.arange(world.n) if ids is None else np.asarray(ids)
        self.ids = base[self.rows] * 2 + side + (int(seed) % 9973) * 1_000_003
        self.member = np.zeros(m, dtype=np.int64) if member is None else np.asarray(member)
        self.hint = hint
        self.cmd = np.zeros((m, ACTIONS[self.game]))
        self.target = None
        self.view = None
        self.count = 0

    def look(self, eyes):
        view = eyes.look(self.delay, self.rows)
        eps = None
        if (self.noise > 0).any():
            eps = hash_normal(self.ids, self.count // self.period, 4) * self.noise[:, None]
        fn = hockey_view if self.game == "hockey" else pong_view
        v = fn(self.w, view, self.rows, self.side, eps)
        # игрок знает, что видит прошлое: до удара осталось меньше на время реакции
        v["ht"] = np.maximum(v["ht"] - self.delay / 240.0, 0.0)
        return v

    def act(self, eyes):
        """Команда стороне в мировой системе: (len(rows), ACTIONS)."""
        if self.count % self.period == 0:
            v = self.look(eyes)
            self.view = v
            if self.policy.kind == "acc":
                vec = hockey_vec(v, self.hint) if self.game == "hockey" else pong_vec(v, self.hint)
                self.cmd = self.policy.forward(vec, self.member)
            else:
                self.target = self.policy.decide(self.game, v)
        self.count += 1
        if self.policy.kind == "acc":
            out = self.cmd.copy()
        else:
            out = self._seek()
        if self.side == 1:
            out[:, 0] = -out[:, 0]
        return out

    def _seek(self):
        """Каждый шаг - к точке бота, по своему состоянию (без задержки)."""
        w, r, s = self.w, self.rows, self.side
        tx, ty, tilt = self.target
        sx = 1.0 if s == 0 else -1.0
        if self.game == "hockey":
            mx = w.mal[r, s, 0] if s == 0 else w.p.X - w.mal[r, s, 0]
            pos = np.stack([mx, w.mal[r, s, 1]], axis=1)
            vel = np.stack([w.mv[r, s, 0] * sx, w.mv[r, s, 1]], axis=1)
            return seek(pos, vel, np.stack([tx, ty], axis=1), w.p.a_max, w.p.v_max, hockey.TICK)
        px = w.pc[r, s, 0] if s == 0 else w.p.L - w.pc[r, s, 0]
        pos = np.stack([px, w.pc[r, s, 1]], axis=1)
        vel = np.stack([w.pv[r, s, 0] * sx, w.pv[r, s, 1]], axis=1)
        a = seek(pos, vel, np.stack([tx, ty], axis=1), w.p.a_max, w.p.v_max, pong.TICK)
        t = seek1(w.th[r, s], w.tw[r, s], tilt, w.p.alpha_max, w.p.w_max, pong.TICK)
        return np.concatenate([a, t[:, None]], axis=1)


def step_all(world, eyes, pilots):
    """Один шаг: глаза записывают, каждый пилот даёт команду своим партиям, физика шагает."""
    see(world, eyes)
    cmd = np.zeros((world.n, 2, ACTIONS[world.game]))
    for pl in pilots:
        cmd[pl.rows, pl.side] = pl.act(eyes)
    world.step(cmd)
