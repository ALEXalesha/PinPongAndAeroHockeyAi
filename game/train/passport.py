"""Паспорт модели: партии до конца против ботов и прошлых уровней.

    python -m game.train passport --game hockey --games 200

Каждая проверка - games партий с зерном seed; партия i: подаёт сторона i % 2, шум глаз -
от номера партии. Итог: доля побед, среднее очков за и против, и первые SAMPLE партий
целиком - tests/test_passport.py пересчитывает их по одной и сверяет.
"""
import json
import os
import time

import numpy as np

from aikit.tournament import summary
from game.ai.match import load_brains, model_path, outcomes, play

SAMPLE = 4
# (проверяемый, соперник): уровень сети или бот; nohint - сеть без подсказки (на «сильном»)
CHECKS = [
    ("strong", "follower"), ("strong", "predictor"), ("strong", "medium"), ("strong", "novice"),
    ("medium", "follower"), ("medium", "predictor"), ("novice", "follower"), ("novice", "predictor"),
    ("nohint", "follower"), ("nohint", "predictor"), ("strong", "nohint"),
]
BOTS = ("follower", "predictor", "idle")


def who(name):
    if name in BOTS:
        return name
    return ("net", name, "strong" if name == "nohint" else name)


def check_seed(i):
    return 1000 + i


def run_check(game, brains, subject, opponent, games, seed, indices=None):
    w = play(game, who(subject), who(opponent), n=games, seed=seed, brains=brains, indices=indices)
    res = outcomes(w)
    return w, res


def make_passport(game, games=200, seed=1, out=None):
    brains = load_brains(game)
    if not brains:
        raise SystemExit(f"нет модели {model_path(game)}")
    with np.load(model_path(game)) as data:
        meta = json.loads(str(data["__meta__"]))
    checks = []
    for i, (subject, opponent) in enumerate(CHECKS):
        if "nohint" in (subject, opponent) and "nohint" not in brains:
            continue
        t0 = time.time()
        w, res = run_check(game, brains, subject, opponent, games, check_seed(i))
        s = summary(res)
        row = {"subject": subject, "opponent": opponent, "games": games, "seed": check_seed(i),
               "wins": s["wins"], "draws": s["draws"], "losses": s["losses"], "win_rate": round(s["wins"] / games, 4),
               "points_for": round(float(w.score[:, 0].mean()), 3),
               "points_against": round(float(w.score[:, 1].mean()), 3),
               "finished": int(w.over.sum()),
               "sample": [[res[k], int(w.score[k, 0]), int(w.score[k, 1])] for k in range(SAMPLE)],
               "seconds": round(time.time() - t0, 1)}
        checks.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    brain = brains["strong"]
    passport = {
        "game": game,
        "network": {"features": int(brain.shape[0]), "hidden": list(brain.shape[1:-1]),
                    "outputs": int(brain.shape[-1]), "parameters": int(brain.params.shape[1])},
        "training": meta,
        "games_per_check": games,
        "checks": checks,
    }
    out = out or model_path(game, "json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(passport, f, ensure_ascii=False, indent=1)
    print(out)
    return passport
