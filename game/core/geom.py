"""Геометрия столкновений для обеих игр: время удара (непрерывное обнаружение) и отскок.

Всё на numpy и с любыми ведущими осями: (n, 2) - n партий сразу, (n, k, 2) - n партий и k
препятствий. Так одна и та же функция считает и окно (n = 1), и обучение (n = сотни).

Непрерывное обнаружение: вместо «сдвинуть и проверить, не залезли ли» ищется момент t
внутри шага, когда расстояние до препятствия станет равно радиусу. Поэтому шайба на любой
скорости не проскакивает сквозь борт или биту: за шаг она может пролететь весь стол, но
удар найдётся в тот момент, когда он случился.
"""
import numpy as np

INF = np.inf


def dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def perp(v):
    """Поворот на +90°: z × v."""
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def norm(v):
    return np.sqrt(dot(v, v))


def clip_norm(v, limit):
    """Вектор не длиннее limit (limit - число или массив по ведущим осям)."""
    n = norm(v)
    scale = np.where(n > limit, limit / np.maximum(n, 1e-300), 1.0)
    return v * scale[..., None]


def toi_point(d, w, radius, tmax):
    """Когда круг (центр p, скорость w относительно точки c) коснётся точки c на расстоянии
    radius. d = p - c. Только сближение. Возвращает t в [0, tmax] или INF.

    Уже внутри и сближаемся - меньший корень отрицательный, и t = 0: такой случай чинится
    отскоком сразу, а не пролётом."""
    a = dot(w, w)
    b = dot(d, w)
    c = dot(d, d) - radius * radius
    disc = b * b - a * c
    approaching = b < 0
    root = (-b - np.sqrt(np.maximum(disc, 0.0))) / np.where(a > 1e-300, a, 1.0)
    t = np.where(approaching & (disc >= 0) & (a > 1e-300), root, INF)
    return np.where(t <= tmax, np.maximum(t, 0.0), INF)


class Segments:
    """Неподвижные отрезки (k, 2, 2): направление, нормаль и длина считаются один раз."""

    def __init__(self, segs):
        segs = np.asarray(segs, dtype=float)
        self.A = segs[..., 0, :]
        e = segs[..., 1, :] - self.A
        self.length = norm(e)
        self.u = e / self.length[..., None]
        self.nn = perp(self.u)

    def toi(self, p, w, radius, tmax):
        """p, w - (n, 2); ответ (t, нормаль) - (n, k) и (n, k, 2)."""
        return _toi_seg(p[:, None, :], w[:, None, :], self.A, self.u, self.nn, self.length, radius, tmax[:, None])


def toi_segment(p, w, A, B, radius, tmax):
    """Когда круг радиуса radius (центр p, скорость w относительно отрезка) коснётся
    отрезка AB боком. Концы отрезка - отдельно, через toi_point. Отрезок двусторонний.

    Возвращает (t, нормаль): t в [0, tmax] или INF; нормаль единичная, от отрезка к кругу."""
    e = B - A
    length = norm(e)
    u = e / length[..., None]
    return _toi_seg(p, w, A, u, perp(u), length, radius, tmax)


def _toi_seg(p, w, A, u, nn, length, radius, tmax):
    d = p - A
    s = d[..., 0] * nn[..., 0] + d[..., 1] * nn[..., 1]
    ds = w[..., 0] * nn[..., 0] + w[..., 1] * nn[..., 1]
    side = np.where(s > 0, 1.0, np.where(s < 0, -1.0, -np.sign(ds)))
    dist = np.abs(s)
    rate = -ds * side                       # > 0 - сближаемся
    ok = rate > 0
    t = np.where(ok, np.maximum(dist - radius, 0.0) / np.where(ok, rate, 1.0), INF)
    tt = np.where(ok, t, 0.0)
    q = (d[..., 0] + w[..., 0] * tt) * u[..., 0] + (d[..., 1] + w[..., 1] * tt) * u[..., 1]
    t = np.where((q >= 0) & (q <= length) & (t <= tmax), t, INF)
    return t, nn * side[..., None]


def contact(v, spin, n, u, e, mu, k, r):
    """Отскок шара от поверхности: v - скорость шара, spin - вращение (рад/с, против
    часовой - плюс), n - нормаль от поверхности к шару, u - скорость поверхности в точке
    касания, e - упругость по нормали, mu - трение, k = I/(m r²) (2/3 у полого шара).

    По нормали скорость сближения меняет знак и умножается на e. По касательной трение
    гасит проскальзывание точки касания, но не больше mu * (импульс по нормали) - закон
    Кулона. Касательный импульс закручивает шар. Возвращает (v, spin) после удара;
    если шар не сближается с поверхностью, ничего не меняется.

    Законы (tests/test_geom.py): при e = 1, mu = 0 скорость относительно поверхности по
    модулю сохраняется; с трением энергия в системе поверхности не растёт."""
    rel = v - u
    vn = dot(rel, n)
    t_ = perp(n)
    vt = dot(rel, t_)
    slip = vt - spin * r
    hit = vn < 0
    jn = np.where(hit, -(1.0 + e) * vn, 0.0)
    jt = -slip / (1.0 + 1.0 / k)
    lim = mu * jn
    jt = np.where(hit, np.clip(jt, -lim, lim), 0.0)
    v_new = v + jn[..., None] * n + jt[..., None] * t_
    spin_new = spin - jt / (k * r)
    return v_new, spin_new
