"""Законы геометрии ударов: время удара - первое касание, отскок - упругость и трение."""
import math

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from game.core.geom import contact, toi_point, toi_segment

coord = st.floats(-3, 3, allow_nan=False)
speed = st.floats(-50, 50, allow_nan=False)
angle = st.floats(0, 2 * math.pi, allow_nan=False)


def v2(x, y):
    return np.array([x, y], dtype=float)


def point_seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-300), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def seg_seg_dist(p0, p1, a, b):
    """Расстояние между отрезками на плоскости: 0, если пересекаются."""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2 = cross(a, b, p0), cross(a, b, p1)
    d3, d4 = cross(p0, p1, a), cross(p0, p1, b)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(point_seg_dist(p0, a, b), point_seg_dist(p1, a, b),
               point_seg_dist(a, p0, p1), point_seg_dist(b, p0, p1))


@given(coord, coord, speed, speed, coord, coord, coord, coord, st.floats(0.01, 0.3))
def test_capsule_toi_is_first_touch(px, py, wx, wy, ax, ay, bx, by, r):
    """Отрезок + оба конца = «капсула». Если удар найден - в этот момент расстояние ровно r,
    а раньше оно не меньше r. Если не найден - путь центра за шаг не подходит ближе r."""
    p, w, A, B = v2(px, py), v2(wx, wy), v2(ax, ay), v2(bx, by)
    assume(np.linalg.norm(B - A) > 1e-3)
    assume(point_seg_dist(p, A, B) > r + 1e-6)
    tmax = 0.5
    ts, _ = toi_segment(p, w, A, B, r, tmax)
    ta = toi_point(p - A, w, r, tmax)
    tb = toi_point(p - B, w, r, tmax)
    t = min(float(ts), float(ta), float(tb))
    if math.isfinite(t):
        assert 0 <= t <= tmax
        assert abs(point_seg_dist(p + w * t, A, B) - r) < 1e-6
        for k in range(40):
            tk = t * k / 40
            assert point_seg_dist(p + w * tk, A, B) >= r - 1e-6
    else:
        assert seg_seg_dist(p, p + w * tmax, A, B) >= r - 1e-6


@given(angle, st.floats(0.01, 0.3), st.floats(0.01, 0.99), st.floats(0.1, 50))
def test_inside_and_approaching_hits_at_once(a, r, depth, sp):
    """Уже внутри и сближаемся - удар сразу (t = 0), а не пролёт насквозь."""
    d = v2(math.cos(a), math.sin(a)) * r * depth
    w = -v2(math.cos(a), math.sin(a)) * sp
    assert float(toi_point(d, w, r, 0.1)) == 0.0
    A, B = v2(-1, 0), v2(1, 0)
    t, nrm = toi_segment(v2(0, r * depth), v2(0, -sp), A, B, r, 0.1)
    assert float(t) == 0.0 and nrm[1] == 1.0


@given(angle, speed, speed, speed, speed, st.floats(-300, 300))
def test_elastic_frictionless_keeps_speed_relative_to_surface(a, vx, vy, ux, uy, spin):
    n = v2(math.cos(a), math.sin(a))
    v, u = v2(vx, vy), v2(ux, uy)
    v2_, s2 = contact(v, np.float64(spin), n, u, 1.0, 0.0, 2 / 3, 0.02)
    assert abs(np.linalg.norm(v2_ - u) - np.linalg.norm(v - u)) < 1e-9 * (1 + np.linalg.norm(v - u))
    assert s2 == spin


@given(angle, speed, speed, speed, speed, st.floats(-300, 300), st.floats(0, 1), st.floats(0, 2))
def test_energy_never_grows(a, vx, vy, ux, uy, spin, e, mu):
    """В системе поверхности энергия шара (поступательная и вращения) после удара не больше."""
    n = v2(math.cos(a), math.sin(a))
    v, u = v2(vx, vy), v2(ux, uy)
    k, r = 2 / 3, 0.02

    def energy(vv, ss):
        rel = vv - u
        return 0.5 * float(rel @ rel) + 0.5 * k * r * r * float(ss) ** 2

    v2_, s2 = contact(v, np.float64(spin), n, u, e, mu, k, r)
    assert energy(v2_, s2) <= energy(v, spin) * (1 + 1e-9) + 1e-12


@given(angle, speed, speed, st.floats(0, 1))
def test_normal_speed_reverses_times_e(a, vx, vy, e):
    n = v2(math.cos(a), math.sin(a))
    v = v2(vx, vy)
    vn = float(v @ n)
    v2_, _ = contact(v, np.float64(0.0), n, v2(0, 0), e, 0.5, 2 / 3, 0.02)
    if vn < 0:
        assert abs(float(v2_ @ n) + e * vn) < 1e-9 * (1 + abs(vn))
    else:
        assert np.array_equal(v2_, v)


@given(angle, speed, speed, st.floats(-300, 300))
def test_big_friction_stops_slipping(a, vx, vy, spin):
    """Трения хватает - точка касания после удара не проскальзывает (шар катится)."""
    n = v2(math.cos(a), math.sin(a))
    v = v2(vx, vy)
    assume(float(v @ n) < -0.1)
    r = 0.02
    v2_, s2 = contact(v, np.float64(spin), n, v2(0, 0), 0.8, 1e6, 2 / 3, r)
    t = v2(-n[1], n[0])
    assert abs(float(v2_ @ t) - float(s2) * r) < 1e-6 * (1 + abs(spin) * r + np.linalg.norm(v))


def test_brushing_up_gives_topspin():
    """Ракетка слева идёт вверх, лицо к сопернику: мяч, летящий влево, получает верхнее
    вращение (по часовой - минус) и подлетает."""
    v, s = contact(v2(-5, 0), np.float64(0.0), v2(1, 0), v2(0, 3), 0.8, 0.6, 2 / 3, 0.02)
    assert s < 0 and v[1] > 0 and v[0] > 0
