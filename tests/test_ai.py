"""Законы честности ИИ и проверки прогноза и ботов.

Честность: ИИ видит шайбу или мяч только через глаза с задержкой, вращения не видит, а
бита или ракетка ИИ двигается той же физикой с теми же пределами, что у человека."""
import math
from dataclasses import replace

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from game.ai import predict
from game.ai.agents import (Brain, Follower, Pilot, Predictor, hash_normal, make_eyes, param_count, see, sizes,
                            step_all)
from game.ai.match import outcomes, play
from game.ai.observe import MAX_DELAY, Eyes
from game.core import hockey, pong
from game.core.hockey import Hockey
from game.core.pong import Pong
from tests.laws import sim


@given(st.lists(st.floats(-5, 5), min_size=1, max_size=200), st.integers(0, MAX_DELAY))
def test_eyes_show_exactly_the_past(values, d):
    eyes = Eyes(1, 1)
    eyes.reset(np.array([[values[0]]]))
    for v in values:
        eyes.push(np.array([[v]]))
    seen = eyes.look(d)[0, 0]
    want = values[-1 - d] if d < len(values) else values[0]
    assert seen == want


def random_brain(game, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return Brain(game, rng.normal(size=param_count(sizes(game))) * scale)


@sim(30)
@given(st.sampled_from(["hockey", "pong"]), st.integers(0, 10 ** 6), st.integers(1, MAX_DELAY))
def test_ai_does_not_see_the_present(game, seed, delay):
    """После того как глаза записали шаг, мяч в мире подменён - решение ИИ то же: ИИ не
    читает мир напрямую, только глаза с задержкой."""
    rng = np.random.default_rng(seed)
    w = Hockey(1, first=0) if game == "hockey" else Pong(1, first=0)
    eyes = make_eyes(w)
    pilot = Pilot(w, 1, random_brain(game, seed), delay=delay, period=1)
    bot = Pilot(w, 1, Predictor(), delay=delay, period=1)
    for _ in range(30):
        step_all(w, eyes, [pilot, bot])
    see(w, eyes)
    a = pilot.act(eyes).copy()
    bot.act(eyes)
    tb = [None if x is None else np.copy(x) for x in bot.target]
    obj = w.puck if game == "hockey" else w.ball
    vel = w.pv if game == "hockey" else w.bv
    obj[:] = rng.uniform(0.2, 0.8, (1, 2))
    vel[:] = rng.normal(size=(1, 2)) * 5
    assert np.array_equal(pilot.act(eyes), a)
    bot.act(eyes)
    for x, y in zip(tb, bot.target):
        assert (x is None and y is None) or np.array_equal(x, y)


def test_ai_does_not_see_spin():
    """Два мира отличаются только вращением мяча - ИИ решает одинаково."""
    outs = []
    for spin in (-200.0, 200.0):
        w = Pong(1, first=0)
        w.place(0, ball=[1.0, 0.3], bv=[-4.0, 1.0], spin=spin, last_hitter=1, bounced=-1)
        eyes = make_eyes(w)
        pilot = Pilot(w, 0, random_brain("pong", 3), delay=10, period=1)
        outs.append(pilot.act(eyes))
    assert np.array_equal(outs[0], outs[1])


@sim(20)
@given(st.sampled_from(["hockey", "pong"]), st.integers(0, 10 ** 6))
def test_crazy_network_obeys_the_same_limits(game, seed):
    """Сеть с огромными весами дёргает изо всех сил - бита всё равно не быстрее v_max."""
    w = Hockey(1, first=0) if game == "hockey" else Pong(1, first=0)
    eyes = make_eyes(w)
    pilots = [Pilot(w, s, random_brain(game, seed + s, scale=50.0), delay=5, period=1) for s in (0, 1)]
    for _ in range(240):
        step_all(w, eyes, pilots)
        v = w.mv if game == "hockey" else w.pv
        assert (np.linalg.norm(v, axis=-1) <= w.p.v_max + 1e-9).all()


def forecast_error(x, y, sp, a, e_wall):
    """|y на линии защиты по физике - прогноз| и |время - прогноз|; None - не долетела."""
    params = replace(hockey.P, R=1e-4, e_wall=e_wall)
    w = Hockey(1, params=params, first=1)
    vx, vy = -sp * math.cos(a), sp * math.sin(a)
    w.place(0, puck=[x, y], pv=[vx, vy], mal=[[params.R, params.R], [params.X - params.R, params.R]])
    hy, t, reach = predict.hockey_intercept(np.array([x]), np.array([y]), np.array([vx]), np.array([vy]), p=params)
    prev = w.puck[0].copy()
    for k in range(240 * 10):
        w.step(np.zeros((1, 2, 2)))
        cur = w.puck[0]
        if cur[0] <= predict.HOCKEY_XD < prev[0]:
            f = (prev[0] - predict.HOCKEY_XD) / (prev[0] - cur[0])
            y_at = prev[1] + f * (cur[1] - prev[1])
            assert reach[0]
            return abs(y_at - hy[0]), abs((k + f) * hockey.TICK - t[0])
        prev = cur.copy()
    assert not reach[0]
    return None


@sim(30)
@given(st.floats(0.4, 1.0), st.floats(0.1, 0.9), st.floats(0.5, 7), st.floats(-0.95, 0.95))
def test_puck_forecast_matches_physics(x, y, sp, a):
    """Шайба без бит на пути и с упругими бортами: прогноз «где пересечёт линию защиты»
    совпадает с физикой - формула с трением и «гармошкой» точная."""
    got = forecast_error(x, y, sp, a, 1.0)
    if got is not None:
        assert got[0] < 5e-3 and got[1] < 0.02


def test_puck_forecast_close_with_real_walls():
    """Настоящие борта теряют 8% скорости поперёк - прогноз этого не знает и ошибается
    на сантиметры, как ошибался бы человек."""
    rng = np.random.default_rng(3)
    errs = [forecast_error(rng.uniform(0.6, 1.0), rng.uniform(0.1, 0.9), rng.uniform(2, 7), rng.uniform(-0.9, 0.9),
                           hockey.P.e_wall) for _ in range(30)]
    errs = [e[0] for e in errs if e is not None]
    assert len(errs) >= 20 and np.median(errs) < 0.03 and max(errs) < 0.12


def test_ball_forecast_is_close_without_spin():
    """Мяч без вращения: прогноз точки удара в пределах нескольких сантиметров от физики.
    Сопротивление воздуха в прогнозе приближено, вращение от отскока учтено (без него
    медиана ошибки была 7 см, с ним 3.5 см; ракетка - 16 см)."""
    rng = np.random.default_rng(4)
    errs = []
    for _ in range(40):
        vx, vy, y0 = -rng.uniform(4, 8), rng.uniform(-0.5, 1.5), rng.uniform(0.15, 0.35)
        w = Pong(1, first=1)
        w.place(0, ball=[2.6, y0], bv=[vx, vy], last_hitter=1, bounced=-1, pc=[[-0.9, 0.85], [3.6, 0.85]])
        hx, hy, t, valid = predict.pong_intercept(np.array([2.6]), np.array([y0]), np.array([vx]), np.array([vy]),
                                                  np.array([False]))
        if not valid[0]:
            continue
        prev = w.ball[0].copy()
        for _ in range(480):
            w.step(np.zeros((1, 2, 3)))
            cur = w.ball[0]
            if hx[0] == predict.PONG_HIT_X and cur[0] <= hx[0] < prev[0]:
                f = (prev[0] - hx[0]) / (prev[0] - cur[0])
                errs.append(abs(prev[1] + f * (cur[1] - prev[1]) - hy[0]))
                break
            prev = cur.copy()
    assert len(errs) >= 10
    assert np.median(errs) < 0.05 and max(errs) < 0.1


def test_hash_noise_is_normal_and_per_game():
    e = hash_normal(np.arange(4000), 3, 4)
    assert abs(e.mean()) < 0.05 and abs(e.std() - 1) < 0.05
    one = hash_normal(np.array([17]), 3, 4)
    assert np.array_equal(one[0], e[17])


def test_match_game_replays_alone():
    """Партия 3 из пачки, пересчитанная одна, проходит так же: паспорт можно проверить."""
    w = play("hockey", "predictor", "follower", 5, seed=9, max_seconds=15)
    one = play("hockey", "predictor", "follower", seed=9, max_seconds=15, indices=[3])
    assert np.array_equal(w.score[3], one.score[0]) and np.array_equal(w.puck[3], one.puck[0])


def test_predictor_beats_follower_in_hockey():
    w = play("hockey", "predictor", "follower", 12, seed=2, max_seconds=40)
    res = outcomes(w)
    assert res.count("W") >= 9, res


def test_predictor_serves_land():
    """Бот-предсказатель подаёт: мяч падает на сторону соперника почти всегда."""
    n = 16
    w = Pong(n, first=np.zeros(n, dtype=int))
    eyes = make_eyes(w)
    pilots = [Pilot(w, 0, Predictor(), delay=30, noise=0.004, period=4), Pilot(w, 1, Follower(), delay=30, period=4)]
    good = np.zeros(n, dtype=bool)
    for _ in range(int(2.0 * 240)):
        step_all(w, eyes, pilots)
        good |= w.good == 0
    assert good.mean() >= 0.9
