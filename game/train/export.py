"""Модель для игры из папок обучения: game/models/<игра>.npz.

Уровни - снимки одного обучения: новичок - ранний снимок (почти копия бота после
подражания), средний - середина, сильный - конец. Плюс nohint - сеть, обученная без
подсказки траектории (для сравнения в паспорте). К уровню в игре добавляются задержка
реакции и ошибка глаза (game/ai/match.py, LEVELS)."""
import json
import os

import numpy as np

from aikit.trainlog import load_snapshot
from game.ai.match import model_path


def pick(snapshots, frac):
    gens = sorted(snapshots)
    want = frac * gens[-1]
    return min(gens, key=lambda g: abs(g - want))


def export(game, run, nohint_run=None, out=None):
    arrays, meta = load_snapshot(os.path.join(run, "state.npz"))
    snaps = {int(k[5:]): v for k, v in arrays.items() if k.startswith("snap_")}
    choice = {"novice": pick(snaps, 0.0), "medium": pick(snaps, 0.5)}
    weights = {k: snaps[g] for k, g in choice.items()}
    weights["strong"] = arrays["mean"]
    info = {"hidden": meta["cfg"]["hidden"], "gens": meta["gen"], "seconds": meta["seconds"],
            "games": meta["games_played"], "imitation": meta.get("imitation", 0),
            "levels_from": {**{k: int(g) for k, g in choice.items()}, "strong": int(meta["gen"])},
            "cfg": meta["cfg"]}
    if nohint_run:
        a2, m2 = load_snapshot(os.path.join(nohint_run, "state.npz"))
        weights["nohint"] = a2["mean"]
        info["nohint"] = {"gens": m2["gen"], "seconds": m2["seconds"], "games": m2["games_played"]}
    out = out or model_path(game)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=np.array(json.dumps(info, ensure_ascii=False)), **weights)
    os.replace(tmp, out)
    print(f"{out}: {', '.join(weights)}")
    return info
