"""Партии пачкой: кто за какую сторону, уровни, итог для паспорта.

Уровень - это снимок обучения плюс задержка реакции и ошибка глаза. Боты для проверок
играют с задержкой BOT_DELAY и малым шумом BOT_NOISE: «средний человек».
"""
import json
import os

import numpy as np

from game.ai.agents import Brain, Follower, Idle, Pilot, Predictor, make_eyes, step_all
from game.core.hockey import Hockey
from game.core.pong import Pong

TICKS = 240
LEVELS = {
    "novice": {"delay": 50, "noise": 0.03},       # 0.21 с
    "medium": {"delay": 34, "noise": 0.015},      # 0.14 с
    "strong": {"delay": 22, "noise": 0.006},      # 0.09 с
}
LEVEL_NAMES = tuple(LEVELS)
BOT_DELAY = 30                                   # 0.125 с
BOT_NOISE = 0.004                                # чуть-чуть: иначе все партии ботов одинаковые
PERIOD = 4                                       # решение 60 раз в секунду
MAX_SECONDS = {"hockey": 300.0, "pong": 600.0}
MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def new_world(game, n, seed=0, first=None):
    return (Hockey if game == "hockey" else Pong)(n, first=first, seed=seed)


def bot(name):
    return {"follower": Follower(), "predictor": Predictor(), "idle": Idle()}[name]


def model_path(game, ext="npz"):
    return os.path.join(MODELS, f"{game}.{ext}")


def load_brains(game, path=None):
    """{имя: Brain} из файла модели: novice, medium, strong, nohint. Нет файла - {}."""
    path = path or model_path(game)
    if not os.path.exists(path):
        return {}
    with np.load(path) as data:
        meta = json.loads(str(data["__meta__"]))
        hidden = tuple(meta["hidden"])
        return {k: Brain(game, data[k], hidden, hint=(k != "nohint")) for k in data.files if k != "__meta__"}


def pilot_for(world, side, who, rows=None, seed=0, brains=None, ids=None):
    """who: 'follower' | 'predictor' | 'idle' | ('net', имя снимка, уровень)."""
    if isinstance(who, str):
        return Pilot(world, side, bot(who), rows=rows, delay=BOT_DELAY, noise=BOT_NOISE, period=PERIOD, seed=seed,
                     ids=ids)
    _, name, level = who
    lv = LEVELS[level]
    brain = brains[name]
    return Pilot(world, side, brain, rows=rows, delay=lv["delay"], noise=lv["noise"], period=PERIOD, seed=seed,
                 hint=brain.hint, ids=ids)


def play(game, who0, who1, n=None, seed=0, max_seconds=None, brains=None, indices=None):
    """Партии до конца (или до max_seconds игрового времени). Партия i зависит только от
    seed и i: подаёт сторона i % 2, шум - от номера партии. Поэтому любую партию можно
    пересчитать одну (indices) и сверить с паспортом. Возвращает мир."""
    idx = np.arange(n) if indices is None else np.asarray(indices)
    w = new_world(game, len(idx), seed=seed, first=idx % 2)
    eyes = make_eyes(w)
    pilots = [pilot_for(w, 0, who0, seed=seed * 2 + 1, brains=brains, ids=idx),
              pilot_for(w, 1, who1, seed=seed * 2 + 2, brains=brains, ids=idx)]
    limit = int((max_seconds or MAX_SECONDS[game]) * TICKS)
    for _ in range(limit):
        step_all(w, eyes, pilots)
        if w.over.all():
            break
    return w


def outcomes(w, side=0):
    """'W'/'L'/'D' по партиям с точки зрения side; недоигранная - по счёту."""
    me, other = w.score[:, side], w.score[:, 1 - side]
    out = np.where(w.over, np.where(w.winner == side, "W", "L"),
                   np.where(me > other, "W", np.where(me < other, "L", "D")))
    return "".join(out.tolist())
