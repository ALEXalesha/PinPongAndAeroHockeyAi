"""Старт обучения: сеть сначала повторяет бота-«предсказателя» (обучение с учителем).

Голый CEM со случайных весов в аэрохоккее за 50 поколений научился касаться шайбы, но
против «предсказателя» играл не лучше стоящей на месте биты (-1.3 гола за 12 с, журнал
09.10). Поэтому сначала сеть учится повторять команды бота по тем же признакам, что видит
сама (с задержкой и шумом), а уже потом CEM в игре с собой и лигой ищет, как играть лучше
бота. Сеть без подсказки учится повторять того же бота, но подсказки не видит - ей надо
самой понять, куда летит шайба. Так сравнение «с подсказкой и без» честное.
"""
import numpy as np

from aikit.nn.layers import Dense, Sequential, Tanh
from aikit.nn.optim import Adam
from game.ai.agents import Pilot, Predictor, hockey_vec, make_eyes, pong_vec, see, sizes
from game.ai.match import BOT_DELAY, BOT_NOISE, LEVELS, PERIOD, TICKS, new_world


COLLECT_GAMES = 64
COLLECT_SECONDS = 40.0
EXAMPLES = int(COLLECT_GAMES * COLLECT_SECONDS * TICKS / PERIOD)      # 153 600 решений бота


def collect(game, n=COLLECT_GAMES, seconds=COLLECT_SECONDS, seed=0, hint=True):
    """Признаки и команды бота: (X, Y). Бот играет за обе стороны с разными задержками."""
    rng = np.random.default_rng(seed)
    w = new_world(game, n, seed=seed, first=rng.integers(0, 2, n))
    eyes = make_eyes(w)
    level = rng.choice(list(LEVELS), n)
    delay = np.array([LEVELS[x]["delay"] for x in level])
    noise = np.array([LEVELS[x]["noise"] for x in level])
    pilots = [Pilot(w, 0, Predictor(), delay=delay, noise=noise, period=PERIOD, seed=seed + 1),
              Pilot(w, 1, Predictor(), delay=BOT_DELAY, noise=BOT_NOISE, period=PERIOD, seed=seed + 2)]
    vec = hockey_vec if game == "hockey" else pong_vec
    A = 2 if game == "hockey" else 3
    xs, ys = [], []
    for k in range(int(seconds * TICKS)):
        see(w, eyes)
        cmd = np.zeros((n, 2, A))
        for side, pl in enumerate(pilots):
            decide = pl.count % pl.period == 0
            out = pl.act(eyes)
            cmd[:, side] = out
            if side == 0 and decide:
                xs.append(vec(pl.view, hint))
                ys.append(out.copy())          # сторона 0 - зеркала нет
        w.step(cmd)
    return np.concatenate(xs), np.concatenate(ys)


def fit(game, X, Y, hidden=(16, 16), epochs=30, batch=256, lr=3e-3, seed=0, log=print):
    """MLP той же формы, что Brain: tanh везде. Возвращает плоский вектор весов."""
    rng = np.random.default_rng(seed)
    shape = sizes(game, hidden)
    layers = []
    for a, b in zip(shape[:-1], shape[1:]):
        layers += [Dense(a, b, rng, dtype=np.float64), Tanh()]
    net = Sequential(layers)
    opt = Adam(net.params(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            j = order[i:i + batch]
            out = net.forward(X[j])
            d = out - Y[j]
            total += float((d * d).sum())
            for _, _, g in net.params():
                g[...] = 0.0
            net.backward(2 * d / len(j))
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"подражание: эпоха {ep + 1}, ошибка {total / n:.4f}")
    flat = []
    for layer in net.items:
        if isinstance(layer, Dense):
            flat += [layer.W.reshape(-1), layer.b.reshape(-1)]
    return np.concatenate(flat)
