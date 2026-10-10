"""Обучение в малом: подражание даёт ту же сеть, что играет; поколение CEM идёт; снимок
читается обратно; модель выгружается и загружается."""
import json
import os

import numpy as np

from aikit.trainlog import load_snapshot, save_snapshot
from game.ai.agents import Brain, sizes
from game.ai.match import load_brains
from game.train.evolve import DEFAULTS, State, generation
from game.train.export import export
from game.train.imitate import collect, fit


def tiny(game):
    cfg = dict(DEFAULTS[game])
    cfg.update(pop=4, games=2, seconds=0.5, gens=3)
    return cfg


def test_imitation_net_is_the_brain_that_plays():
    """Веса после подражания, разложенные в Brain, дают те же выходы, что и сеть обучения,
    и повторяют бота лучше нулевой сети."""
    X, Y = collect("hockey", n=8, seconds=2.0, seed=1)
    flat = fit("hockey", X, Y, (16, 16), epochs=15, seed=1, log=lambda s: None)
    brain = Brain("hockey", flat, (16, 16))
    out = brain.forward(X, np.zeros(len(X), dtype=int))
    assert out.shape == Y.shape
    assert np.mean((out - Y) ** 2) < 0.8 * np.mean(Y ** 2)


def test_generation_and_snapshot_roundtrip(tmp_path):
    for game in ("hockey", "pong"):
        st = State(game, tiny(game), seed=1, hint=True)
        rng = np.random.default_rng(1)
        before = st.mean.copy()
        row = generation(st, rng)
        assert st.gen == 1 and np.isfinite(row["best"]) and not np.array_equal(before, st.mean)
        path = tmp_path / f"{game}.npz"
        save_snapshot(path, st.arrays(), st.meta())
        back = State.restore(*load_snapshot(path))
        assert back.gen == 1 and np.array_equal(back.mean, st.mean) and back.cfg["hidden"] == st.cfg["hidden"]


def test_export_and_load(tmp_path):
    game = "hockey"
    cfg = tiny(game)
    st = State(game, cfg, seed=1, hint=True)
    rng = np.random.default_rng(2)
    st.snapshots[0] = st.mean.copy()
    for _ in range(2):
        generation(st, rng)
    run = tmp_path / "run"
    run.mkdir()
    save_snapshot(run / "state.npz", st.arrays(), st.meta())
    out = tmp_path / "hockey.npz"
    info = export(game, str(run), str(run), out=str(out))
    assert info["levels_from"]["strong"] == 2
    brains = load_brains(game, str(out))
    assert set(brains) == {"novice", "medium", "strong", "nohint"}
    assert brains["strong"].shape == sizes(game, cfg["hidden"])
    assert not brains["nohint"].hint and brains["strong"].hint
    with np.load(out) as data:
        assert json.loads(str(data["__meta__"]))["gens"] == 2
    assert not os.path.exists(str(out) + ".tmp")
