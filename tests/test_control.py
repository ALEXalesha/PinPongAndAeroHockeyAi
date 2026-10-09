"""Часы и мышь: шагов за одно и то же время столько же при любой частоте кадров, партия от
частоты кадров не зависит; бита за мышью доезжает и останавливается в пределах."""
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from game.core.control import Clock, seek
from game.core.hockey import P, TICK, Hockey

frames = st.lists(st.floats(0.0, 0.05), min_size=1, max_size=200)


@given(frames, st.integers(0, 10 ** 6))
def test_clock_counts_steps_from_total_time(chunks, seed):
    """Те же 1-2 секунды, порезанные на кадры двух разных частот, дают одно число шагов."""
    total = sum(chunks)
    a = Clock(TICK, max_steps=10 ** 9)
    for c in chunks:
        a.feed(c)
    rng = np.random.default_rng(seed)
    b = Clock(TICK, max_steps=10 ** 9)
    left = total
    while left > 1e-12:
        c = min(left, float(rng.uniform(0, 0.03)))
        b.feed(c)
        left -= c
    assert abs(a.steps - b.steps) <= 1
    assert abs(a.steps - total / TICK) <= 1


def test_clock_drops_backlog_after_long_freeze():
    c = Clock(TICK, max_steps=24)
    assert c.feed(5.0) == 24
    assert c.feed(TICK) == 1


def play(frame_times):
    """Партия, где команды зависят только от номера шага, при заданных временах кадров."""
    h = Hockey(1, first=0)
    clock = Clock(TICK, max_steps=10 ** 9)
    for ft in frame_times:
        for _ in range(clock.feed(ft)):
            k = h.ticks
            cmd = np.array([[[np.sin(k * 0.05), np.cos(k * 0.031)], [np.cos(k * 0.02), np.sin(k * 0.07)]]])
            h.step(cmd)
    return h


def test_game_does_not_depend_on_fps():
    a = play([1 / 30] * 90)                 # 3 с по 30 кадров
    b = play([1 / 144] * 432)               # 3 с по 144 кадра
    n = min(a.ticks, b.ticks)
    assert abs(a.ticks - b.ticks) <= 1 and n >= 719
    # одинаковое число шагов - одинаковая партия, бит в бит
    c = play([1 / 30] * 90)
    assert np.array_equal(a.puck, c.puck) and np.array_equal(a.mal, c.mal)
    if a.ticks == b.ticks:
        assert np.array_equal(a.puck, b.puck) and np.array_equal(a.mal, b.mal)


def test_mallet_follows_mouse_and_stops():
    """300 случайных точек сразу (пачкой): бита за 1.2 с доезжает и стоит."""
    n = 300
    rng = np.random.default_rng(11)
    target = np.stack([rng.uniform(P.R, P.X / 2 - P.R, n), rng.uniform(P.R, P.Y - P.R, n)], axis=1)
    h = Hockey(n, first=1)
    for _ in range(int(1.2 / TICK)):
        cmd = np.zeros((n, 2, 2))
        cmd[:, 0] = seek(h.mal[:, 0], h.mv[:, 0], target, P.a_max, P.v_max, TICK)
        assert (np.linalg.norm(cmd[:, 0], axis=-1) <= 1 + 1e-9).all()
        h.step(cmd)
    assert (np.linalg.norm(h.mal[:, 0] - target, axis=-1) < 0.01).all()
    assert (np.linalg.norm(h.mv[:, 0], axis=-1) < 0.2).all()
