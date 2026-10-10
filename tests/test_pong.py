"""Законы пинг-понга: мяч не проходит сквозь стол, сетку и ракетку; дуга, отскок и
вращение ведут себя как в жизни; правила очка и подачи."""
import math
from dataclasses import replace

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tests.laws import sim

from game.core.pong import PAUSE, PLAY, SERVE, P, TICK, Pong, paddle_box, server_for

ZERO = np.zeros((1, 2, 3))
FAR = [[-0.9, 0.8], [P.L + 0.9, 0.8]]          # ракетки далеко, не мешают


def solo(params=P, **kw):
    g = Pong(1, params=params, first=0)
    kw.setdefault("pc", FAR)
    g.place(0, **kw)
    return g


def run(g, ticks, until=None):
    for _ in range(ticks):
        g.step(ZERO)
        if until is not None and until(g):
            return True
    return False


@sim()
@given(st.floats(0.1, P.L - 0.1), st.floats(0.03, 0.5), st.floats(1, 3000), st.floats(-0.5, 0.5))
def test_ball_never_passes_through_table(x, h, sp, slope):
    """Мяч бьёт в стол сверху с любой скоростью - отскакивает, не оказываясь под столом."""
    g = solo(ball=[x, h], bv=[sp * slope, -sp], last_hitter=1)
    for _ in range(3):
        g.step(ZERO)
        bx, by = g.ball[0]
        if 0 <= bx <= P.L:
            assert by >= P.r - 1e-9


STILL_AIR = replace(P, g=0.0, drag=0.0, magnus=0.0)       # мяч летит прямо - бьёт точно в цель


@sim()
@given(st.floats(P.r + 0.005, P.net_h - 0.01), st.floats(1, 3000), st.floats(0.05, 0.6))
def test_ball_never_passes_through_net(h, sp, dist):
    g = solo(STILL_AIR, ball=[P.L / 2 - dist, h], bv=[sp, 0.0], last_hitter=0)
    for _ in range(200):
        g.step(ZERO)
        assert g.ball[0, 0] <= P.L / 2 - P.r + 1e-9
        if g.net[0]:
            break
    assert g.net[0] and g.bv[0, 0] < 0


@sim()
@given(st.floats(-0.06, 0.06), st.floats(1, 100), st.floats(0.05, 0.4), st.floats(-0.8, 0.8))
def test_ball_never_passes_through_paddle(offset, sp, dist, th):
    """Мяч летит в стоящую ракетку до 100 м/с (втрое быстрее рекордного удара) - ракетка его
    отбивает. На сотнях м/с мяч за шаг успевает отскочить от ракетки, от стола и обогнуть
    край ракетки - это уже не пролёт, а несколько честных ударов за 1/240 с."""
    c = np.array([-0.3, 0.4])
    g = solo(STILL_AIR, ball=[c[0] + dist, c[1] + offset * math.cos(th)], bv=[-sp, 0.0], pc=[c, FAR[1]],
             th=[th, 0.0], last_hitter=1, bounced=0)
    for _ in range(200):
        g.step(ZERO)
        assert np.dot(g.ball[0] - c, [math.cos(th), math.sin(th)]) > 0     # с той же стороны лица
        if g.hit[0, 0]:
            break
    assert g.hit[0, 0]


def test_free_flight_is_a_parabola_without_air():
    params = replace(P, drag=0.0, magnus=0.0)
    g = solo(params, ball=[-0.5, 0.5], bv=[3.0, 2.0], last_hitter=0)
    t = 0.3
    run(g, round(t / TICK))
    assert abs(g.ball[0, 0] - (-0.5 + 3.0 * t)) < 1e-9
    assert abs(g.ball[0, 1] - (0.5 + 2.0 * t - 0.5 * P.g * t * t)) < 0.02


def landing_x(spin):
    g = solo(ball=[0.3, 0.3], bv=[5.0, 1.0], spin=spin, last_hitter=0)
    run(g, 240, until=lambda g: g.bounce[0] >= 0)
    return g.ball[0, 0]


def test_topspin_dives_backspin_floats():
    """Эффект Магнуса: верхнее вращение (по часовой, минус) прижимает мяч - он падает
    раньше; нижнее - держит в воздухе."""
    top, flat, back = landing_x(-150.0), landing_x(0.0), landing_x(150.0)
    assert top < flat < back


def bounce_vx(spin):
    g = solo(replace(P, magnus=0.0), ball=[1.8, 0.05], bv=[3.0, -2.0], spin=spin, last_hitter=0)
    run(g, 60, until=lambda g: g.bounce[0] >= 0)
    return g.bv[0, 0]


def test_topspin_kicks_forward_on_bounce():
    """Верхнее вращение толкает мяч вперёд при отскоке, нижнее - не толкает (трение
    по Кулону: на сильном нижнем мяч тормозится так же, как без вращения)."""
    assert bounce_vx(-300.0) > bounce_vx(0.0) >= bounce_vx(300.0)


def test_drop_without_losses_comes_back_up():
    params = replace(P, drag=0.0, magnus=0.0, e_table=1.0, mu_table=0.0)
    g = solo(params, ball=[0.7, 0.5], bv=[0.0, 0.0], last_hitter=1)
    top = 0.0
    bounced = False
    for _ in range(int(0.9 / TICK)):
        g.step(ZERO)
        bounced |= g.bounce[0] >= 0
        if bounced:
            top = max(top, g.ball[0, 1])
    assert bounced and abs(top - 0.5) < 0.01


@sim()
@given(st.lists(st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)), min_size=1, max_size=60))
def test_paddle_limits(cmds):
    g = Pong(1, first=0)
    box = paddle_box()
    for c in cmds:
        v0, w0 = g.pv.copy(), g.tw.copy()
        g.step(np.array([[[c[0] * 5, c[1] * 5, c[2] * 3], c]]))
        assert (np.linalg.norm(g.pv, axis=-1) <= P.v_max + 1e-9).all()
        assert (np.linalg.norm(g.pv - v0, axis=-1) <= P.a_max * TICK + 1e-9).all()
        assert (np.abs(g.tw) <= P.w_max + 1e-9).all()
        assert (np.abs(g.tw - w0) <= P.alpha_max * TICK + 1e-9).all()
        assert (np.abs(g.th) <= P.tilt_max + 1e-12).all()
        for side in (0, 1):
            x, y = g.pc[0, side]
            assert box[side, 0] - 1e-12 <= x <= box[side, 1] + 1e-12
            assert box[side, 2] - 1e-12 <= y <= box[side, 3] + 1e-12


def test_batch_equals_solo():
    rng = np.random.default_rng(7)
    n = 5
    cmds = rng.uniform(-1, 1, (700, n, 2, 3))
    batch = Pong(n, first=np.arange(n) % 2)
    for c in cmds:
        batch.step(c)
    for i in range(n):
        one = Pong(1, first=i % 2)
        for c in cmds:
            one.step(c[i:i + 1])
        assert np.array_equal(one.ball[0], batch.ball[i])
        assert np.array_equal(one.pc[0], batch.pc[i])
        assert np.array_equal(one.score[0], batch.score[i])


# --- правила ---


def test_ball_on_own_side_is_a_fault():
    g = solo(ball=[0.5, 0.2], bv=[0.0, -1.0], last_hitter=0, bounced=-1)
    assert run(g, 120, until=lambda g: g.point[0] >= 0)
    assert g.fault[0] == 0 and g.point[0] == 1 and g.score[0].tolist() == [0, 1]


def test_good_shot_then_second_bounce_wins_the_point():
    g = solo(ball=[1.9, 0.2], bv=[0.5, -1.0], last_hitter=0, bounced=-1)
    run(g, 240, until=lambda g: g.point[0] >= 0)
    assert g.point[0] == 0 and g.fault[0] == 1


def test_shot_off_the_table_loses():
    g = solo(ball=[1.9, 0.5], bv=[8.0, 1.0], last_hitter=0, bounced=-1)
    run(g, 480, until=lambda g: g.point[0] >= 0)
    assert g.point[0] == 1 and g.fault[0] == 0


def test_volley_is_a_fault():
    """Ракетка встречает мяч до отскока на своей стороне - очко сопернику."""
    c = np.array([-0.2, 0.3])
    g = solo(ball=[0.0, 0.3], bv=[-3.0, 0.0], pc=[c, FAR[1]], last_hitter=1, bounced=-1)
    run(g, 60, until=lambda g: g.point[0] >= 0)
    assert g.fault[0] == 0 and g.point[0] == 1


def test_missed_toss_gives_point_to_receiver():
    g = Pong(1, first=0)
    g.pc[0] = FAR
    run(g, int(3.0 / TICK), until=lambda g: g.point[0] >= 0)
    assert g.point[0] == 1 and g.fault[0] == 0
    assert g.phase[0] == PAUSE
    run(g, int(P.pause / TICK) + 2)
    assert g.phase[0] in (SERVE, PLAY)


def test_serve_rotation():
    first = 0
    seq = [int(server_for(first, k)) for k in range(24)]
    assert seq[:8] == [0, 0, 1, 1, 0, 0, 1, 1]
    assert seq[18:24] == [1, 1, 0, 1, 0, 1]           # с 10:10 - каждое очко


def test_game_needs_two_point_lead():
    g = solo(ball=[1.9, 0.2], bv=[0.5, -1.0], last_hitter=0, bounced=-1)
    g.score[0] = [10, 10]
    run(g, 240, until=lambda g: g.point[0] >= 0)
    assert g.score[0].tolist() == [11, 10] and not g.over[0]
    g.place(0, ball=[1.9, 0.2], bv=[0.5, -1.0], last_hitter=0, bounced=-1, pc=FAR)
    run(g, 240, until=lambda g: g.point[0] >= 0)
    assert g.score[0].tolist() == [12, 10] and g.over[0] and g.winner[0] == 0


def test_toss_goes_up_after_hold():
    g = Pong(1, first=1)
    assert g.phase[0] == SERVE and g.ball[0, 0] > P.L
    run(g, int(P.hold / TICK) + 3)
    assert g.phase[0] == PLAY and g.bv[0, 1] > 0


def test_moving_paddle_adds_speed():
    """Ракетка, идущая вперёд, отправляет мяч быстрее, чем стоящая."""
    def after(pv):
        c = np.array([-0.3, 0.4])
        g = solo(STILL_AIR, ball=[-0.2, 0.4], bv=[-2.0, 0.0], pc=[c, FAR[1]], pv=[[pv, 0.0], [0.0, 0.0]],
                 last_hitter=1, bounced=0)
        for _ in range(60):
            g.step(np.array([[[1.0 if pv else 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))
            if g.hit[0, 0]:
                return g.bv[0, 0]
        raise AssertionError("нет удара")
    assert after(4.0) > after(0.0) + 4.0


def serve_from(strict, bv):
    """Подача стороны 0 с руки: мяч только что отбит подающим."""
    g = solo(ball=[-0.1, 0.25], bv=bv, last_hitter=0, bounced=-1)
    g.stroke[0] = 1
    g.strict[0, 0] = strict
    run(g, 480, until=lambda g: g.point[0] >= 0 or g.good[0] >= 0)
    return g


def test_simple_serve_lands_on_receiver_side():
    g = serve_from(False, [6.0, 1.5])
    assert g.good[0] == 0 and g.point[0] == -1


def test_strict_serve_needs_own_bounce_first():
    """Строгая подача прямо на сторону соперника - ошибка подающего."""
    g = serve_from(True, [6.0, 1.5])
    assert g.point[0] == 1 and g.fault[0] == 0


def test_strict_serve_with_own_bounce_is_good():
    g = serve_from(True, [5.0, -0.3])
    assert g.serve_own[0]
    assert g.good[0] == 0 and g.point[0] == -1
