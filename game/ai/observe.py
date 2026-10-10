"""Что видит игрок: шайбу или мяч с задержкой реакции и с ошибкой глаза.

Глаза (Eyes) каждый шаг записывают положение и скорость шайбы или мяча в кольцевой буфер.
ИИ читает из буфера состояние delay шагов назад - это и есть задержка реакции. Сверху -
шум: по положению sigma, по скорости sigma * 8 (скорость на глаз оценивается хуже).
Вращение мяча в буфер не пишется вовсе: его не видит человек, не видит и ИИ.

Своя бита или ракетка ИИ видна без задержки - это ощущение своей руки, а не глаз.
Соперник - с той же задержкой, что и мяч.

Всё зеркалится для стороны 1: ИИ всегда «стоит слева», одна сеть играет за обе стороны.
"""
import numpy as np

MAX_DELAY = 72          # 0.3 с при 240 шагах в секунду


class Eyes:
    """Буфер на n партий: (n, MAX_DELAY + 1, k) - шайба/мяч (x, y, vx, vy) и биты обеих сторон."""

    def __init__(self, n, width):
        self.n = n
        self.buf = np.zeros((n, MAX_DELAY + 1, width))
        self.pos = 0
        self.filled = 0

    def push(self, row):
        """row (n, width) - что видно на этом шаге."""
        self.pos = (self.pos + 1) % (MAX_DELAY + 1)
        self.buf[:, self.pos] = row
        self.filled = min(self.filled + 1, MAX_DELAY + 1)

    def reset(self, row):
        self.buf[:] = row[:, None, :]
        self.filled = MAX_DELAY + 1

    def look(self, delay, rows=None):
        """Что видно с задержкой delay шагов - не старше заполненного. rows - какие партии
        (по умолчанию все), delay - число или массив той же длины."""
        rows = np.arange(self.n) if rows is None else np.asarray(rows)
        d = np.minimum(np.broadcast_to(np.asarray(delay, dtype=np.int64), rows.shape), self.filled - 1)
        idx = (self.pos - d) % (MAX_DELAY + 1)
        return self.buf[rows, idx]


def hockey_row(w):
    """Что видно в аэрохоккее: шайба (4), биты (2 × 4)."""
    return np.concatenate([w.puck, w.pv, w.mal.reshape(w.n, 4), w.mv.reshape(w.n, 4)], axis=1)


def pong_row(w):
    """Что видно в пинг-понге: мяч (4) без вращения, ракетки (2 × 2 центр), наклоны (2), фаза,
    подающий, кто бил последним и где был отскок - это видно глазами и на табло."""
    return np.concatenate([w.ball, w.bv, w.pc.reshape(w.n, 4), w.th,
                           w.phase[:, None].astype(float), w.server[:, None].astype(float),
                           w.last_hitter[:, None].astype(float), w.bounced[:, None].astype(float)], axis=1)


def mirror_x(x, length, side):
    """Координата x глазами стороны side: у стороны 1 поле отражено."""
    return np.where(side == 0, x, length - x)


def mirror_v(vx, side):
    return np.where(side == 0, vx, -vx)
