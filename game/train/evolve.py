"""Обучение: метод перекрёстной энтропии (CEM) в игре с собой, лига прошлых версий, боты.

Поколение: из нормального распределения (mean, sigma) берутся P наборов весов. Каждый
играет G коротких партий (seconds игрового времени): с ботом-«предсказателем», со
«следящим» и с прошлыми версиями себя из лиги - чтобы не забывать старые приёмы. Половину
партий - слева, половину - справа. Задержка реакции и ошибка глаза в каждой партии - свои,
от сильного до новичка: сеть учится играть при любой.

Оценка - разница голов (очков) за партию плюс небольшая награда за касания и удачные
удары (в пинг-понге - мяч на стороне соперника). Лучшая четверть задаёт новые mean и sigma.
Каждые league_every поколений mean уходит в лигу и в снимки уровней.

Всё поколение - одна пачка партий numpy в одном процессе с одним потоком BLAS.
"""
import time
from dataclasses import replace

import numpy as np

from game.ai.agents import Brain, Follower, Pilot, Predictor, make_eyes, param_count, sizes, step_all
from game.ai.match import BOT_DELAY, BOT_NOISE, LEVELS, PERIOD, TICKS, new_world

DEFAULTS = {
    "hockey": dict(pop=32, games=8, seconds=12.0, gens=120, hidden=(16, 16), sigma=0.08, hit_bonus=0.05,
                   good_bonus=0.0),
    "pong": dict(pop=32, games=8, seconds=16.0, gens=120, hidden=(32, 32), sigma=0.08, hit_bonus=0.05,
                 good_bonus=0.3),
}
# соперники в G партиях одного члена популяции: 'predictor', 'follower', 'league'
SCHEDULE = ("predictor", "predictor", "predictor", "follower", "league", "league", "league", "league")


class State:
    """Всё, что нужно для продолжения с места: распределение, лига, история."""

    def __init__(self, game, cfg, seed, hint):
        self.game, self.cfg, self.seed, self.hint = game, cfg, seed, hint
        self.dim = param_count(sizes(game, cfg["hidden"]))
        self.mean = np.zeros(self.dim)
        self.sigma = np.full(self.dim, cfg["sigma"])
        self.gen = 0
        self.league = [self.mean.copy()]
        self.snapshots = {}
        self.history = []
        self.seconds = 0.0
        self.games_played = 0
        self.imitation = 0

    def arrays(self):
        a = {"mean": self.mean, "sigma": self.sigma, "league": np.array(self.league)}
        for k, v in self.snapshots.items():
            a["snap_" + str(k)] = v
        return a

    def meta(self):
        return {"game": self.game, "cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in self.cfg.items()},
                "seed": self.seed, "hint": self.hint, "gen": self.gen, "history": self.history,
                "seconds": self.seconds, "games_played": self.games_played,
                "imitation": self.imitation}

    @classmethod
    def restore(cls, arrays, meta):
        cfg = dict(meta["cfg"])
        cfg["hidden"] = tuple(cfg["hidden"])
        st = cls(meta["game"], cfg, meta["seed"], meta["hint"])
        st.mean, st.sigma = arrays["mean"], arrays["sigma"]
        st.league = list(arrays["league"])
        st.snapshots = {int(k[5:]): v for k, v in arrays.items() if k.startswith("snap_")}
        st.gen, st.history = meta["gen"], meta["history"]
        st.seconds, st.games_played = meta["seconds"], meta["games_played"]
        st.imitation = meta.get("imitation", 0)
        return st


def evaluate(game, params, cfg, league, rng, hint=True):
    """Оценка P наборов весов: (fitness (P,), сводка)."""
    P, G = len(params), cfg["games"]
    n = P * G
    member = np.repeat(np.arange(P), G)
    slot = np.tile(np.arange(G), P)
    # одинаковые условия для всех членов популяции (общие случайные числа): кто подаёт,
    # уровень, соперник из лиги - разница оценок тогда от весов, а не от везения
    first = np.tile(rng.integers(0, 2, G), P)
    w = new_world(game, n, seed=int(rng.integers(2 ** 31)), first=first)
    w.p = replace(w.p, to_win=10 ** 6)                  # короткая партия без конца
    eyes = make_eyes(w)
    me_side = (slot % 2).astype(np.int64)
    level = np.tile(rng.choice(list(LEVELS), G), P)
    delay = np.array([LEVELS[x]["delay"] for x in level])
    noise = np.array([LEVELS[x]["noise"] for x in level])
    brain = Brain(game, params, cfg["hidden"], hint=hint)
    league_brain = Brain(game, np.array(league), cfg["hidden"], hint=hint)
    pick = np.tile(rng.integers(0, len(league), G), P)
    pilots = []
    for side in (0, 1):
        mine = np.nonzero(me_side == side)[0]
        pilots.append(Pilot(w, side, brain, rows=mine, delay=delay[mine], noise=noise[mine], period=PERIOD,
                            seed=int(rng.integers(2 ** 31)), member=member[mine], hint=hint))
        theirs = np.nonzero(me_side != side)[0]
        kinds = np.array([SCHEDULE[s % len(SCHEDULE)] for s in slot[theirs]])
        for kind in ("predictor", "follower"):
            rows = theirs[kinds == kind]
            if len(rows):
                pol = Predictor() if kind == "predictor" else Follower()
                pilots.append(Pilot(w, side, pol, rows=rows, delay=BOT_DELAY, noise=BOT_NOISE, period=PERIOD,
                                    seed=int(rng.integers(2 ** 31))))
        rows = theirs[kinds == "league"]
        if len(rows):
            pilots.append(Pilot(w, side, league_brain, rows=rows, delay=delay[rows], noise=noise[rows],
                                period=PERIOD, seed=int(rng.integers(2 ** 31)), member=pick[rows], hint=hint))
    hits = np.zeros(n)
    good = np.zeros(n)
    idx = np.arange(n)
    for _ in range(int(cfg["seconds"] * TICKS)):
        step_all(w, eyes, pilots)
        hits += w.hit[idx, me_side]
        if game == "pong":
            good += w.good == me_side
    gf = w.score[idx, me_side]
    ga = w.score[idx, 1 - me_side]
    raw = (gf - ga).astype(float)
    score = raw + cfg["hit_bonus"] * np.minimum(hits, 20) + cfg["good_bonus"] * np.minimum(good, 20)
    fit = score.reshape(P, G).mean(axis=1)
    kinds_all = np.array([SCHEDULE[s % len(SCHEDULE)] for s in slot])
    summary = {k: float(raw[kinds_all == k].mean()) for k in ("predictor", "follower", "league")}
    summary["hits"] = float(hits.mean())
    if game == "pong":
        summary["good"] = float(good.mean())
    return fit, summary


def generation(st, rng):
    cfg = st.cfg
    P = cfg["pop"]
    eps = rng.normal(size=(P, st.dim))
    eps[0] = 0.0                                   # сам mean тоже в популяции - видно, растёт ли он
    params = st.mean + eps * st.sigma
    t0 = time.time()
    fit, summary = evaluate(st.game, params, cfg, st.league, rng, hint=st.hint)
    elite = np.argsort(-fit)[: max(2, P // 4)]
    st.mean = params[elite].mean(axis=0)
    # sigma по элите плюс медленно убывающий шум: CEM без него быстро «замерзает»
    extra = cfg["sigma"] * 0.2 * max(0.05, 1 - st.gen / max(cfg["gens"], 1))
    st.sigma = np.sqrt(params[elite].var(axis=0) + extra ** 2)
    st.gen += 1
    st.seconds += time.time() - t0
    st.games_played += P * cfg["games"]
    every = max(1, cfg["gens"] // 12)
    if st.gen % every == 0:
        st.league.append(st.mean.copy())
        st.league = st.league[-8:]
    if st.gen % every == 0 or st.gen == cfg["gens"]:
        st.snapshots[st.gen] = st.mean.copy()
    row = {"gen": st.gen, "best": float(fit.max()), "mean_member": float(fit[0]), "avg": float(fit.mean()),
           "sigma": float(st.sigma.mean()), "sec": round(time.time() - t0, 2), **{k: round(v, 3) for k, v in
                                                                                     summary.items()}}
    st.history.append(row)
    return row
