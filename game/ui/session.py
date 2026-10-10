"""Партия в окне: мир на одну партию, игроки сторон, часы. Без Qt - проверяется отдельно.

Стороны: 'mouse' (человек мышью), 'keys' (второй человек на клавиатуре), 'ai' (сеть уровня
или бот, если модели нет). Человек ведёт биту/ракетку к точке через seek - с теми же
пределами скорости и ускорения, что у ИИ.
"""
import numpy as np

from game.ai import predict
from game.ai.agents import Pilot, Predictor, make_eyes, see
from game.ai.match import LEVELS, PERIOD, load_brains, new_world
from game.core import hockey, pong
from game.core.control import Clock, seek, seek1

KEY_SPEED = {"hockey": 2.2, "pong": 1.8}         # как быстро клавиши двигают точку, ед/с
TILT_STEP = 0.1
TILT_KEY_SPEED = 2.5


class Human:
    """Точка, куда человек ведёт биту или ракетку, и наклон (пинг-понг)."""

    def __init__(self, world, side):
        self.side = side
        if world.game == "hockey":
            self.target = world.mal[0, side].copy()
        else:
            self.target = world.pc[0, side].copy()
        self.tilt = 0.25
        self.keys = set()          # 'left', 'right', 'up', 'down', 'tilt+', 'tilt-'

    def clamp(self, world):
        box = world.box[self.side]
        self.target = np.clip(self.target, box[[0, 2]], box[[1, 3]])
        lim = getattr(world.p, "tilt_max", 1.0)
        self.tilt = float(np.clip(self.tilt, -lim, lim))

    def tick_keys(self, world, dt):
        if not self.keys:
            return
        sp = KEY_SPEED[world.game] * dt
        d = np.array([("right" in self.keys) - ("left" in self.keys), ("up" in self.keys) - ("down" in self.keys)],
                     dtype=float)
        if world.game == "hockey":
            d[1] = -d[1]           # на экране аэрохоккея y растёт вниз
        self.target = self.target + d * sp
        self.tilt += (("tilt+" in self.keys) - ("tilt-" in self.keys)) * TILT_KEY_SPEED * dt
        self.clamp(world)

    def command(self, world):
        s = self.side
        if world.game == "hockey":
            return seek(world.mal[0, s], world.mv[0, s], self.target, world.p.a_max, world.p.v_max, hockey.TICK)
        a = seek(world.pc[0, s], world.pv[0, s], self.target, world.p.a_max, world.p.v_max, pong.TICK)
        t = seek1(world.th[0, s], world.tw[0, s], self.tilt, world.p.alpha_max, world.p.w_max, pong.TICK)
        return np.array([a[0], a[1], t])


class MatchSession:
    def __init__(self, game, sides, level="medium", seed=0, first=0, strict_serve=False):
        self.game = game
        self.level = level
        self.world = new_world(game, 1, seed=seed, first=[first])
        self.eyes = make_eyes(self.world)
        self.clock = Clock(hockey.TICK)
        self.sides = list(sides)
        self.brains = load_brains(game)
        self.humans = {}
        self.pilots = {}
        for s, kind in enumerate(self.sides):
            if kind in ("mouse", "keys"):
                self.humans[s] = Human(self.world, s)
            else:
                self.pilots[s] = self._ai(s, seed)
        self.events = []
        if game == "pong" and strict_serve:
            for side in self.humans:           # ИИ учился на простой подаче - строгая только людям
                self.world.strict[0, side] = True

    def _ai(self, side, seed):
        lv = LEVELS[self.level]
        brain = self.brains.get(self.level)
        policy = brain if brain is not None else Predictor()
        return Pilot(self.world, side, policy, delay=lv["delay"], noise=lv["noise"], period=PERIOD,
                     seed=seed * 7 + side, hint=True)

    @property
    def uses_net(self):
        return bool(self.brains)

    def advance(self, seconds):
        """Прошло seconds настоящего времени: сделать нужное число шагов. События - в events."""
        steps = self.clock.feed(seconds)
        for _ in range(steps):
            self.step()
        return steps

    def simulate(self, seconds):
        for _ in range(int(round(seconds / hockey.TICK))):
            self.step()

    def step(self):
        w = self.world
        if w.over[0]:
            return
        dt = hockey.TICK
        see(w, self.eyes)
        A = 2 if self.game == "hockey" else 3
        cmd = np.zeros((1, 2, A))
        for s, h in self.humans.items():
            h.tick_keys(w, dt)
            cmd[0, s] = h.command(w)
        for s, p in self.pilots.items():
            cmd[0, s] = p.act(self.eyes)[0]
        w.step(cmd)
        if w.hit[0].any():
            self.events.append("hit")
        if self.game == "hockey":
            if w.goal[0] >= 0:
                self.events.append(("goal", int(w.goal[0])))
        else:
            if w.point[0] >= 0:
                self.events.append(("point", int(w.point[0])))
            if w.bounce[0] >= 0:
                self.events.append("bounce")
        if w.over[0]:
            self.events.append(("over", int(w.winner[0])))

    def take_events(self):
        ev, self.events = self.events, []
        return ev

    # --- «Мысли ИИ» в мировых координатах ---

    def thoughts(self, side):
        """(ломаная прогноза, точка удара) для ИИ стороны side или None."""
        p = self.pilots.get(side)
        if p is None or p.view is None:
            return None
        v = p.view
        L = self.world.p.X if self.game == "hockey" else self.world.p.L

        def wx(x):
            return x if side == 0 else L - x
        if self.game == "hockey":
            path = predict.hockey_path(float(v["px"][0]), float(v["py"][0]), float(v["pvx"][0]), float(v["pvy"][0]),
                                       p=self.world.p)
            aim = (float(v["hx"][0]), float(v["hy"][0])) if v["valid"][0] else None
            if p.target is not None:
                aim = (float(p.target[0][0]), float(p.target[1][0]))
        else:
            path = predict.pong_path(float(v["bx"][0]), float(v["by"][0]), float(v["bvx"][0]), float(v["bvy"][0]),
                                     bool(v["bounced_mine"][0]), p=self.world.p)
            aim = (float(v["hx"][0]), float(v["hy"][0])) if v["valid"][0] else None
        path = [(wx(x), y) for x, y in path]
        if aim is not None:
            aim = (wx(aim[0]), aim[1])
        return path, aim
