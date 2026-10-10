"""Паспорт модели: все проверки на месте, и первые партии пересчитываются по одной с теми же
зёрнами - счёт в счёт. Партия до конца долгая, поэтому пересчёт - для двух проверок."""
import json
import os

import pytest

from game.ai.match import load_brains, model_path
from game.train.passport import CHECKS, SAMPLE, run_check

REPLAY = [("strong", "follower"), ("strong", "predictor")]


def passport(game):
    path = model_path(game, "json")
    if not os.path.exists(path):
        pytest.skip(f"{game}: модель ещё не обучена")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("game", ["hockey", "pong"])
def test_passport_has_every_check_and_training_facts(game):
    pp = passport(game)
    have = {(c["subject"], c["opponent"]) for c in pp["checks"]}
    assert have == set(CHECKS)
    for c in pp["checks"]:
        assert c["games"] == pp["games_per_check"] == 200
        assert c["wins"] + c["draws"] + c["losses"] == c["games"]
        assert len(c["sample"]) == SAMPLE
    tr = pp["training"]
    assert tr["gens"] > 0 and tr["games"] > 0 and tr["seconds"] > 0 and tr["imitation"] > 0
    brain = load_brains(game)["strong"]
    assert pp["network"]["parameters"] == brain.params.shape[1]


@pytest.mark.slow
@pytest.mark.parametrize("game", ["hockey", "pong"])
@pytest.mark.parametrize("subject,opponent", REPLAY)
def test_sample_replays_with_same_seeds(game, subject, opponent):
    pp = passport(game)
    c = next(c for c in pp["checks"] if c["subject"] == subject and c["opponent"] == opponent)
    w, res = run_check(game, load_brains(game), subject, opponent, None, c["seed"], indices=range(SAMPLE))
    got = [[res[k], int(w.score[k, 0]), int(w.score[k, 1])] for k in range(SAMPLE)]
    assert got == c["sample"]
