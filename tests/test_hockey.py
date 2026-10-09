"""Законы аэрохоккея: шайба не проходит сквозь борт и биту, отражение от борта, пределы
биты одни для всех, партии в пачке считаются так же, как поодиночке, правила счёта."""
import math
from dataclasses import replace

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tests.laws import sim

from game.core.hockey import P, TICK, Hockey, home, mallet_box, spawn

ZERO = np.zeros((1, 2, 2))


def solo(params=P, **kw):
    h = Hockey(1, params=params, first=0)
    h.place(0, **kw)
    return h


@sim()
@given(st.floats(P.r, P.X - P.r), st.floats(P.r, P.Y - P.r), st.floats(0, 2 * math.pi),
       st.floats(1, 5000), st.lists(st.tuples(st.floats(-1, 1), st.floats(-1, 1)), min_size=8, max_size=8))
def test_puck_never_leaves_rink(x, y, a, sp, cmds):
    """На любой скорости, хоть 5000 ед/с (поле - 2 ед), шайба остаётся в поле или влетает
    в ворота - тогда засчитывается гол."""
    h = solo(puck=[x, y], pv=[sp * math.cos(a), sp * math.sin(a)], mal=[[0.1, 0.1], [1.9, 0.9]])
    for c in cmds:
        h.step(np.array([[c, c]]))
        if h.goal[0] >= 0:
            break
        px, py = h.puck[0]
        assert P.r - 1e-9 <= py <= P.Y - P.r + 1e-9
        in_mouth = P.goal_lo < py < P.goal_hi
        assert in_mouth or P.r - 1e-9 <= px <= P.X - P.r + 1e-9


@sim()
@given(st.floats(-0.04, 0.04), st.floats(1, 3000), st.floats(0.1, 0.3))
def test_puck_never_passes_through_mallet(offset, sp, dist):
    """Шайба летит прямо в стоящую биту с любой скоростью - отскакивает назад, а не
    оказывается за битой."""
    m = np.array([0.6, 0.5])
    h = solo(puck=[m[0] + dist, m[1] + offset], pv=[-sp, 0.0], mal=[m, [1.8, 0.5]])
    for _ in range(80):
        h.step(ZERO)
        assert np.linalg.norm(h.puck[0] - m) >= P.r + P.R - 1e-9
        if h.hit[0, 0]:
            break
    assert h.hit[0, 0], "шайба не ударилась о биту"


@sim()
@given(st.floats(0.3, 1.7), st.floats(0.5, 6), st.floats(0.2, 1.4))
def test_wall_reflection_angle(x, sp, a):
    """Без трения и с упругостью 1: угол падения равен углу отражения, скорость та же."""
    params = replace(P, damping=0.0, e_wall=1.0)
    vx, vy = sp * math.cos(a), sp * math.sin(a)
    h = solo(params, puck=[x, P.Y - P.r - 1e-5], pv=[vx, vy], mal=[[0.1, 0.1], [1.9, 0.1]])
    h.step(ZERO)
    assert h.wall[0]
    assert abs(h.pv[0, 0] - vx) < 1e-12 and abs(h.pv[0, 1] + vy) < 1e-9


@sim()
@given(st.floats(0.5, 6), st.floats(-0.07, 0.07))
def test_elastic_mallet_hit_keeps_speed(sp, offset):
    """Упругий удар о неподвижную биту: скорость шайбы по модулю сохраняется."""
    params = replace(P, damping=0.0, e_mallet=1.0)
    h = solo(params, puck=[0.75, 0.5 + offset], pv=[-sp, 0.0], mal=[[0.6, 0.5], [1.8, 0.5]])
    for _ in range(int(0.5 / TICK)):
        h.step(ZERO)
        if h.hit[0, 0]:
            break
    assert h.hit[0, 0]
    assert abs(np.linalg.norm(h.pv[0]) - sp) < 1e-9


def test_damping_slows_puck():
    h = solo(puck=[1.0, 0.5], pv=[0.5, 0.0])
    for _ in range(240):
        h.step(ZERO)
    assert abs(h.pv[0, 0] - 0.5 * math.exp(-P.damping)) < 1e-9


@sim()
@given(st.lists(st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)),
                min_size=1, max_size=60))
def test_mallet_limits_same_for_everyone(cmds):
    h = Hockey(1, first=0)
    box = mallet_box()
    for c in cmds:
        v0 = h.mv.copy()
        h.step(np.array([[[c[0] * 9, c[1] * 9], [c[2], c[3]]]]))
        assert (np.linalg.norm(h.mv, axis=-1) <= P.v_max + 1e-9).all()
        assert (np.linalg.norm(h.mv - v0, axis=-1) <= P.a_max * TICK + 1e-9).all()
        for side in (0, 1):
            x, y = h.mal[0, side]
            assert box[side, 0] - 1e-12 <= x <= box[side, 1] + 1e-12
            assert box[side, 2] - 1e-12 <= y <= box[side, 3] + 1e-12


def test_batch_equals_solo():
    """Партия в пачке из многих идёт шаг в шаг так же, как одна: окно и обучение - одна физика."""
    rng = np.random.default_rng(5)
    n = 6
    cmds = rng.uniform(-1, 1, (900, n, 2, 2))
    batch = Hockey(n, seed=3)
    batch.pv[:] = rng.normal(size=(n, 2)) * 3
    starts = [(batch.puck[i].copy(), batch.pv[i].copy()) for i in range(n)]
    for c in cmds:
        batch.step(c)
    for i in range(n):
        one = Hockey(1, first=0)
        one.place(0, puck=starts[i][0], pv=starts[i][1])
        for c in cmds:
            one.step(c[i:i + 1])
        assert np.array_equal(one.puck[0], batch.puck[i])
        assert np.array_equal(one.mal[0], batch.mal[i])
        assert np.array_equal(one.score[0], batch.score[i])


@sim()
@given(st.floats(0.3, 1.7), st.floats(0.1, 0.9), st.floats(-3, 3), st.floats(-3, 3),
       st.lists(st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)),
                min_size=1, max_size=40))
def test_mirror_symmetry(x, y, vx, vy, cmds):
    """Отражённое поле и отражённые команды дают отражённую партию: ИИ один на обе стороны."""
    a = solo(puck=[x, y], pv=[vx, vy])
    b = solo(puck=[P.X - x, y], pv=[-vx, vy])
    for c in cmds:
        ca = np.array([[[c[0], c[1]], [c[2], c[3]]]])
        cb = np.array([[[-c[2], c[3]], [-c[0], c[1]]]])
        a.step(ca)
        b.step(cb)
        assert abs(a.puck[0, 0] - (P.X - b.puck[0, 0])) < 1e-7 and abs(a.puck[0, 1] - b.puck[0, 1]) < 1e-7
        assert abs(a.mal[0, 0, 0] - (P.X - b.mal[0, 1, 0])) < 1e-7
        assert a.score[0, 0] == b.score[0, 1] and a.score[0, 1] == b.score[0, 0]


def test_goal_scores_and_puck_goes_to_conceder():
    h = solo(puck=[0.3, 0.5], pv=[-4.0, 0.0], mal=[[0.1, 0.1], [1.9, 0.1]])
    for _ in range(60):
        h.step(ZERO)
        if h.goal[0] >= 0:
            break
    assert h.goal[0] == 1 and h.score[0].tolist() == [0, 1]
    assert np.array_equal(h.puck[0], spawn(0)) and np.array_equal(h.mal[0], home())


def test_post_bounces_puck_back():
    h = solo(puck=[0.3, P.goal_lo], pv=[-4.0, 0.0], mal=[[0.1, 0.9], [1.9, 0.1]])
    for _ in range(60):
        h.step(ZERO)
    assert h.score[0].tolist() == [0, 0] and h.pv[0, 0] > 0


def test_stall_rule_gives_puck_to_opponent():
    h = solo(puck=[0.5, 0.2], pv=[0.0, 0.0], mal=[[0.1, 0.9], [1.9, 0.1]])
    for _ in range(int(P.stall / TICK) + 2):
        h.step(ZERO)
    assert np.array_equal(h.puck[0], spawn(1))
    assert h.score[0].tolist() == [0, 0]


def test_match_ends_at_seven():
    h = solo(puck=[0.3, 0.5])
    h.score[0] = [0, P.to_win - 1]
    h.place(0, pv=[-4.0, 0.0], mal=[[0.1, 0.1], [1.9, 0.1]])
    for _ in range(60):
        h.step(ZERO)
    assert h.over[0] and h.winner[0] == 1
    frozen = h.snapshot()
    h.step(np.ones((1, 2, 2)))
    assert np.array_equal(h.snapshot()["puck"], frozen["puck"])
