"""Пинг-понг, вид сбоку. Своя 2D-физика с шагом 1/240 с, n партий сразу.

Стол - отрезок y = 0 от x = 0 до x = L, сетка - вертикальный отрезок посередине. Мяч летит
по дуге: тяжесть, сопротивление воздуха и эффект Магнуса от вращения. Отскок от стола и
ракетки - с трением (game/core/geom.contact): удар скользящей ракеткой закручивает мяч, а
закрученный мяч по-другому отскакивает от стола. Сторона 0 - слева, сторона 1 - справа.

Ракетка - отрезок длиной 2*HALF с центром и наклоном. Наклон th: 0 - лицо смотрит прямо на
соперника, плюс - лицо открыто вверх. Управление - ускорением, как у биты в аэрохоккее:
команда (n, 2, 3) - (ax, ay) не длиннее 1 в долях a_max и угловое ускорение в [-1, 1].

Правила (упрощены, см. docs/spec.md, «решения»):
- подача: мяч подброшен у своего края стола, подающий бьёт, мяч должен упасть на сторону
  соперника. Простая подача - сразу на сторону соперника; строгая (strict[side] = True) -
  как по правилам: сначала отскок на своей стороне, потом на стороне соперника;
- после удара мяч должен один раз отскочить на стороне соперника, потом соперник бьёт;
- ошибка: свой стол, мимо стола, удар до отскока, второй отскок у принимающего - его ошибка;
- подача меняется каждые 2 очка, с 10:10 - каждое очко; партия до 11 с разницей в 2.
"""
from dataclasses import dataclass

import numpy as np

from game.core.geom import Segments, clip_norm, contact, norm, perp, toi_point, toi_segment

TICK = 1.0 / 240.0
SERVE, PLAY, PAUSE = 0, 1, 2


@dataclass(frozen=True)
class Params:
    L: float = 2.74                 # длина стола
    net_h: float = 0.1525
    r: float = 0.02                 # мяч
    g: float = 9.81
    drag: float = 0.12              # сопротивление: a = -drag |v| v
    magnus: float = 0.004           # a = magnus * spin * (z × v)
    spin_decay: float = 0.3         # 1/с
    floor: float = -0.76            # пол ниже стола
    e_table: float = 0.88
    mu_table: float = 0.25
    e_net: float = 0.15
    mu_net: float = 0.3
    e_paddle: float = 0.8
    mu_paddle: float = 0.6
    k: float = 2.0 / 3.0            # I / (m r²) полого шара
    half: float = 0.08              # половина ракетки
    v_max: float = 6.0
    a_max: float = 60.0
    w_max: float = 14.0             # наклон: рад/с
    alpha_max: float = 200.0        # наклон: рад/с²
    tilt_max: float = 1.2
    reach: float = 1.0              # ракетка отходит от края стола не дальше
    over: float = 0.1               # и заходит над столом не дальше
    low: float = -0.3               # ниже стола (за краем)
    high: float = 0.9
    over_table_y: float = 0.08      # над столом ракетка не ниже
    toss_x: float = 0.15            # подброс: за краем стола
    toss_y: float = 0.22
    toss_v: float = 2.2
    hold: float = 0.8               # сколько мяч лежит перед подбросом
    pause: float = 0.7              # пауза после очка
    rally_max: float = 30.0         # розыгрыш дольше - переигровка
    same_stroke: float = 0.1        # касания одной ракеткой в пределах - один удар
    to_win: int = 11
    substeps: int = 4


P = Params()


def paddle_box(p=P):
    """(2 стороны, [xmin, xmax, ymin, ymax]) для центра ракетки."""
    return np.array([[-p.reach, p.over, p.low, p.high],
                     [p.L - p.over, p.L + p.reach, p.low, p.high]])


def paddle_home(p=P):
    return np.array([[-0.3, 0.2], [p.L + 0.3, 0.2]])


def normal(th, side):
    """Нормаль лица ракетки стороны side: к сопернику и вверх на угол th."""
    sx = np.where(np.asarray(side) == 0, 1.0, -1.0)
    return np.stack([sx * np.cos(th), np.sin(th)], axis=-1)


def server_for(first, points, p=P):
    """Кто подаёт после points разыгранных очков: смена каждые 2, с 10:10 - каждое."""
    deuce = 2 * (p.to_win - 1)
    k = np.where(points < deuce, points // 2, (p.to_win - 1) + (points - deuce))
    return (first + k) % 2


class Pong:
    game = "pong"

    def __init__(self, n=1, params=P, first=None, seed=0):
        self.p = params
        self.n = n
        rng = np.random.default_rng(seed)
        self.first = (rng.integers(0, 2, n) if first is None
                      else np.broadcast_to(np.asarray(first), (n,)).astype(np.int64).copy())
        self.table = Segments([[[0.0, 0.0], [params.L, 0.0]],
                               [[params.L / 2, 0.0], [params.L / 2, params.net_h]]])
        self.corners = np.array([[0.0, 0.0], [params.L, 0.0], [params.L / 2, params.net_h]])
        self.box = paddle_box(params)
        self.pc = np.repeat(paddle_home(params)[None], n, axis=0)
        self.pv = np.zeros((n, 2, 2))
        self.th = np.zeros((n, 2))
        self.tw = np.zeros((n, 2))
        self.ball = np.zeros((n, 2))
        self.bv = np.zeros((n, 2))
        self.spin = np.zeros(n)
        self.score = np.zeros((n, 2), dtype=np.int64)
        self.over = np.zeros(n, dtype=bool)
        self.winner = np.full(n, -1)
        self.server = self.first.copy()
        self.phase = np.full(n, SERVE)
        self.phase_t = np.zeros(n)
        self.last_hitter = np.full(n, -1)
        self.bounced = np.full(n, -1)          # сторона отскока после последнего удара
        self.hit_t = np.zeros(n)               # время с последнего удара
        self.stroke = np.zeros(n, dtype=np.int64)   # ударов в розыгрыше (1 - летит подача)
        self.serve_own = np.zeros(n, dtype=bool)    # строгая подача уже отскочила на своей
        self.strict = np.zeros((n, 2), dtype=bool)  # строгая подача у стороны
        self.ticks = 0
        self._events()
        self._hold(np.ones(n, dtype=bool))

    def _events(self):
        self.hit = np.zeros((self.n, 2), dtype=bool)
        self.bounce = np.full(self.n, -1)       # сторона стола, куда отскочил мяч на этом шаге
        self.net = np.zeros(self.n, dtype=bool)
        self.point = np.full(self.n, -1)        # кто выиграл очко на этом шаге
        self.fault = np.full(self.n, -1)        # чья ошибка
        self.good = np.full(self.n, -1)         # чей удар долетел до стола соперника

    def toss_point(self, side):
        p = self.p
        x = np.where(side == 0, -p.toss_x, p.L + p.toss_x)
        return np.stack([x, np.full(np.shape(side), p.toss_y)], axis=-1)

    def _hold(self, m):
        self.ball[m] = self.toss_point(self.server[m])
        self.bv[m] = 0.0
        self.spin[m] = 0.0
        self.phase[m] = SERVE
        self.phase_t[m] = 0.0
        self.last_hitter[m] = -1
        self.bounced[m] = -1
        self.stroke[m] = 0
        self.serve_own[m] = False

    # --- ракетки ---

    def move_paddles(self, cmd):
        p = self.p
        cmd = np.asarray(cmd, dtype=float)
        acc = clip_norm(cmd[..., :2], 1.0) * p.a_max
        v = clip_norm(self.pv + acc * TICK, p.v_max)
        c0 = self.pc
        c1 = c0 + v * TICK
        lo = np.broadcast_to(self.box[None, :, [0, 2]], c1.shape).copy()
        hi = self.box[None, :, [1, 3]]
        over_table = np.stack([c1[:, 0, 0] > -p.half, c1[:, 1, 0] < p.L + p.half], axis=1)
        lo[..., 1] = np.where(over_table, p.over_table_y, lo[..., 1])
        clamped = np.clip(c1, lo, hi)
        self.pv = np.where(clamped != c1, 0.0, v)
        self.pc = clamped
        w = np.clip(self.tw + np.clip(cmd[..., 2], -1, 1) * p.alpha_max * TICK, -p.w_max, p.w_max)
        th = self.th + w * TICK
        th_c = np.clip(th, -p.tilt_max, p.tilt_max)
        self.tw = np.where(th_c != th, 0.0, w)
        self.th = th_c
        return c0, clamped, (clamped - c0) / TICK

    def paddle_ends(self, c, th):
        """Концы ракеток (n, 2, 2 конца, 2) по центрам и наклонам."""
        n = normal(th, np.array([0, 1]))
        u = perp(n) * self.p.half
        return c - u, c + u

    # --- шаг ---

    def step(self, cmd):
        p = self.p
        self._events()
        live = ~self.over
        cmd = np.where(live[:, None, None], cmd, 0.0)
        c0, c1, vp = self.move_paddles(cmd)
        self.phase_t += TICK
        self.hit_t += TICK
        # подача: мяч лежит, потом подлетает
        serve = live & (self.phase == SERVE)
        toss = serve & (self.phase_t >= p.hold)
        if toss.any():
            self.phase[toss] = PLAY
            self.phase_t[toss] = 0.0
            self.bv[toss] = [0.0, p.toss_v]
        flying = live & (self.phase != SERVE)
        v = self.bv
        speed = norm(v)
        acc = (np.array([0.0, -p.g]) - p.drag * speed[:, None] * v
               + p.magnus * self.spin[:, None] * perp(v))
        v = np.where(flying[:, None], v + acc * TICK, v)
        self.spin = np.where(flying, self.spin * np.exp(-p.spin_decay * TICK), self.spin)
        pos = self.ball.copy()
        a0, b0 = self.paddle_ends(c0, self.th)
        rows = np.nonzero(flying)[0]
        tau = np.full(len(rows), TICK)
        s = np.zeros(len(rows))
        for _ in range(p.substeps):
            if len(rows) == 0:
                break
            pr, vr = pos[rows], v[rows]
            ts, ns = self.table.toi(pr, vr, p.r, tau)                                       # (k, 2)
            tc = toi_point(pr[:, None, :] - self.corners[None], vr[:, None, :], p.r, tau[:, None])  # (k, 3)
            off = vp[rows] * s[:, None, None]
            A, B = a0[rows] + off, b0[rows] + off
            w = vr[:, None, :] - vp[rows]
            tpd, npd = toi_segment(pr[:, None, :], w, A, B, p.r, tau[:, None])              # (k, 2)
            te = toi_point(pr[:, None, None, :] - np.stack([A, B], axis=2), w[:, :, None, :], p.r,
                           tau[:, None, None]).reshape(len(rows), 4)                         # (k, 4)
            allt = np.concatenate([ts, tc, tpd, te], axis=1)
            which = allt.argmin(axis=1)
            k = np.arange(len(rows))
            tmin = allt[k, which]
            hit = np.isfinite(tmin)
            adv = np.where(hit, tmin, tau)
            pr = pr + vr * adv[:, None]
            s = s + adv
            tau = tau - adv
            pos[rows] = pr
            if not hit.any():
                break
            h = np.nonzero(hit)[0]
            wh = which[h]
            g = rows[h]
            nrm = np.zeros((len(h), 2))
            surf = np.zeros((len(h), 2))
            e = np.full(len(h), p.e_table)
            mu = np.full(len(h), p.mu_table)
            # стол и сетка
            m = wh < 2
            nrm[m] = ns[h[m], wh[m]]
            net = (wh == 1) | (wh == 4)
            e[net], mu[net] = p.e_net, p.mu_net
            m = (wh >= 2) & (wh < 5)
            if m.any():
                d = pr[h[m]] - self.corners[wh[m] - 2]
                nrm[m] = d / np.maximum(norm(d), 1e-12)[:, None]
            # ракетки: лицо (5, 6) и концы (7..10)
            pad = wh >= 5
            side = np.where(wh < 7, wh - 5, (wh - 7) // 2)
            m = (wh >= 5) & (wh < 7)
            nrm[m] = npd[h[m], wh[m] - 5]
            m = wh >= 7
            if m.any():
                ends = np.stack([A, B], axis=2)[h[m], side[m], (wh[m] - 7) % 2]
                d = pr[h[m]] - ends
                nrm[m] = d / np.maximum(norm(d), 1e-12)[:, None]
            if pad.any():
                sd = side[pad]
                gp = g[pad]
                ctr = c0[gp, sd] + vp[gp, sd] * s[h[pad], None]
                cp = pr[h[pad]] - nrm[pad] * p.r - ctr
                surf[pad] = vp[gp, sd] + self.tw[gp, sd][:, None] * perp(cp)
                e[pad], mu[pad] = p.e_paddle, p.mu_paddle
            vn, sn = contact(vr[h], self.spin[g], nrm, surf, e, mu, p.k, p.r)
            v[g] = vn
            self.spin[g] = sn
            # события - по правилам
            table_hit = (wh == 0) | (wh == 2) | (wh == 3)
            if table_hit.any():
                gt = g[table_hit]
                self._on_bounce(gt, (pr[h[table_hit], 0] >= p.L / 2).astype(np.int64))
            self.net[g[net]] = True
            if pad.any():
                self._on_hit(g[pad], side[pad])
            rows, tau, s = g, tau[h], s[h]
        self.ball = np.where(flying[:, None], pos, self.ball)
        self.bv = np.where(flying[:, None], v, self.bv)
        self.ticks += 1
        self._after(live)

    # --- правила ---

    def _award(self, g, winner):
        """Очко стороне winner в партиях g (только идущие розыгрыши)."""
        m = self.phase[g] == PLAY
        g, winner = g[m], winner[m]
        if len(g) == 0:
            return
        self.score[g, winner] += 1
        self.point[g] = winner
        self.phase[g] = PAUSE
        self.phase_t[g] = 0.0

    def _fault(self, g, side):
        m = self.phase[g] == PLAY
        self.fault[g[m]] = side[m]
        self._award(g, 1 - side)

    def _on_hit(self, g, side):
        live = self.phase[g] == PLAY
        g, side = g[live], side[live]
        if len(g) == 0:
            return
        self.hit[g, side] = True
        last = self.last_hitter[g]
        same = (last == side) & (self.hit_t[g] < self.p.same_stroke)
        g, side, last = g[~same], side[~same], last[~same]
        toss_wrong = (last == -1) & (side != self.server[g])
        double = (last == side)
        volley = (last >= 0) & (last != side) & (self.bounced[g] != side)
        bad = toss_wrong | double | volley
        if bad.any():
            self._fault(g[bad], side[bad])
        ok = ~bad
        self.last_hitter[g[ok]] = side[ok]
        self.bounced[g[ok]] = -1
        self.hit_t[g[ok]] = 0.0
        self.stroke[g[ok]] += 1

    def _on_bounce(self, g, side):
        live = self.phase[g] == PLAY
        g, side = g[live], side[live]
        if len(g) == 0:
            return
        self.bounce[g] = side
        last = self.last_hitter[g]
        toss = last == -1
        if toss.any():
            self._fault(g[toss], self.server[g[toss]])
        # строгая подача: первый отскок - на своей стороне, второй - у соперника
        strict = (last >= 0) & (self.stroke[g] == 1) & self.strict[g, np.maximum(last, 0)]
        own_ok = strict & (side == last) & ~self.serve_own[g]
        skipped = strict & (side != last) & ~self.serve_own[g]
        self.serve_own[g[own_ok]] = True
        if skipped.any():
            self._fault(g[skipped], last[skipped])
        own = (last >= 0) & (side == last) & ~own_ok
        if own.any():
            self._fault(g[own], last[own])
        second = (last >= 0) & (side != last) & (self.bounced[g] == side) & ~skipped
        if second.any():
            self._fault(g[second], side[second])
        first = (last >= 0) & (side != last) & (self.bounced[g] != side) & ~skipped
        self.bounced[g[first]] = side[first]
        self.good[g[first]] = last[first]

    def _after(self, live):
        p = self.p
        play = live & (self.phase == PLAY)
        x, y = self.ball[:, 0], self.ball[:, 1]
        down = play & ((y - p.r <= p.floor) | (x < -3.0) | (x > p.L + 3.0))
        if down.any():
            g = np.nonzero(down)[0]
            last = self.last_hitter[g]
            toss = last == -1
            if toss.any():
                self._fault(g[toss], self.server[g[toss]])
            ok = (~toss) & (self.bounced[g] == 1 - last)
            if ok.any():
                self._award(g[ok], last[ok])
            bad = (~toss) & ~ok
            if bad.any():
                self._fault(g[bad], last[bad])
        let = play & (self.phase == PLAY) & (self.phase_t >= p.rally_max)
        if let.any():
            self._hold(let)
        won = live & (self.score.max(axis=1) >= p.to_win) & \
            (np.abs(self.score[:, 0] - self.score[:, 1]) >= 2)
        if won.any():
            self.over |= won
            self.winner = np.where(won, self.score.argmax(axis=1), self.winner)
        nxt = live & ~won & (self.phase == PAUSE) & (self.phase_t >= p.pause)
        if nxt.any():
            self.server[nxt] = server_for(self.first[nxt], self.score[nxt].sum(axis=1), p)
            self._hold(nxt)

    # --- для окна и проверок ---

    def place(self, i, ball=None, bv=None, spin=None, pc=None, pv=None, th=None, phase=PLAY, last_hitter=None,
              bounced=None):
        if ball is not None:
            self.ball[i] = ball
        if bv is not None:
            self.bv[i] = bv
        if spin is not None:
            self.spin[i] = spin
        if pc is not None:
            self.pc[i] = pc
        if pv is not None:
            self.pv[i] = pv
        if th is not None:
            self.th[i] = th
        self.phase[i] = phase
        self.phase_t[i] = 0.0
        if last_hitter is not None:
            self.last_hitter[i] = last_hitter
            self.hit_t[i] = 1.0
        if bounced is not None:
            self.bounced[i] = bounced

    def snapshot(self, i=0):
        return {"ball": self.ball[i].copy(), "bv": self.bv[i].copy(), "spin": float(self.spin[i]),
                "pc": self.pc[i].copy(), "th": self.th[i].copy(), "score": self.score[i].copy(),
                "phase": int(self.phase[i]), "server": int(self.server[i]), "over": bool(self.over[i])}
