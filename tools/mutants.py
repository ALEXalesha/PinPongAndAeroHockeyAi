# Мутанты для tools/mutate.py: (имя, файл, было, стало, проверки).
G = "game/core/geom.py"
TG = "tests/test_geom.py"
H = "game/core/hockey.py"
TH = "tests/test_hockey.py"
PO = "game/core/pong.py"
TP = "tests/test_pong.py"
C = "game/core/control.py"
TC = "tests/test_control.py"
MUTANTS = [
    ("geom: упругость не учитывается", G, "jn = np.where(hit, -(1.0 + e) * vn, 0.0)",
     "jn = np.where(hit, -2.0 * vn, 0.0)", TG),
    ("geom: трение без предела Кулона", G, "jt = np.where(hit, np.clip(jt, -lim, lim), 0.0)",
     "jt = np.where(hit, jt, 0.0)", TG),
    ("geom: вращение не в ту сторону", G, "spin_new = spin - jt / (k * r)", "spin_new = spin + jt / (k * r)", TG),
    ("geom: внутри - не удар", G, "return np.where(t <= tmax, np.maximum(t, 0.0), INF)",
     "return np.where(t <= tmax, t, INF)", TG),
    ("geom: отрезок бесконечный", G, "t = np.where((q >= 0) & (q <= length) & (t <= tmax), t, INF)",
     "t = np.where(t <= tmax, t, INF)", TG),
    ("geom: без радиуса", G, "np.maximum(dist - radius, 0.0)", "np.maximum(dist, 0.0)", TG),
    ("hockey: 7 секунд - 14", H, "stalled = live & (scorer < 0) & (self.stall_t >= p.stall)",
     "stalled = live & (scorer < 0) & (self.stall_t >= 2 * p.stall)", TH),
    ("hockey: бита без предела скорости", H, "v = clip_norm(self.mv + acc * TICK, p.v_max)",
     "v = self.mv + acc * TICK", TH),
    ("hockey: ворота закрыты бортом", H, "t[:, 2:] = np.where(mouth, INF, t[:, 2:])", "pass", TH),
    ("hockey: гол не тому", H, "scorer = np.where(x < 0, 1, np.where(x > p.X, 0, -1))",
     "scorer = np.where(x < 0, 0, np.where(x > p.X, 1, -1))", TH),
    ("hockey: до 8", H, "won = live & (self.score.max(axis=1) >= p.to_win)",
     "won = live & (self.score.max(axis=1) > p.to_win)", TH),
    ("hockey: бита неподвижна в ударе", H, "surf[km] = vmr[hm, side]", "surf[km] = 0.0", TH),
    ("hockey: без трения шайбы", H, "vel = self.pv * np.exp(-p.damping * TICK)", "vel = self.pv * 1.0", TH),
    ("hockey: штанги не мешают", H, "nrm[kp] = d / norm(d)[:, None]", "nrm[kp] = -d / norm(d)[:, None]", TH),
    ("pong: Магнус наоборот", PO, "+ p.magnus * self.spin[:, None] * perp(v))",
     "- p.magnus * self.spin[:, None] * perp(v))", TP),
    ("pong: без тяжести", PO, "acc = (np.array([0.0, -p.g])", "acc = (np.array([0.0, 0.0])", TP),
    ("pong: свой стол не ошибка", PO, "own = (last >= 0) & (side == last)", "own = (last >= 0) & (side == last) & False", TP),
    ("pong: удар с лёта можно", PO, "volley = (last >= 0) & (last != side) & (self.bounced[g] != side)",
     "volley = last < -5", TP),
    ("pong: второй отскок не ошибка", PO, "second = (last >= 0) & (side != last) & (self.bounced[g] == side)",
     "second = (last >= 0) & (side != last) & (self.bounced[g] == side) & False", TP),
    ("pong: мимо стола засчитан", PO, "ok = (~toss) & (self.bounced[g] == 1 - last)", "ok = ~toss", TP),
    ("pong: подача по 2 и после 10:10", PO, "k = np.where(points < deuce, points // 2, (p.to_win - 1) + (points - deuce))",
     "k = points // 2", TP),
    ("pong: разница в 1", PO, "(np.abs(self.score[:, 0] - self.score[:, 1]) >= 2)",
     "(np.abs(self.score[:, 0] - self.score[:, 1]) >= 1)", TP),
    ("pong: ракетка без предела скорости", PO, "v = clip_norm(self.pv + acc * TICK, p.v_max)",
     "v = self.pv + acc * TICK", TP),
    ("pong: ракетка неподвижна в ударе", PO, "surf[pad] = vp[gp, sd] + self.tw[gp, sd][:, None] * perp(cp)",
     "surf[pad] = 0.0", TP),
    ("pong: подброса нет", PO, "self.bv[toss] = [0.0, p.toss_v]", "self.bv[toss] = [0.0, 0.0]", TP),
    ("control: шаги по кускам", C, "k = int(self.total / self.tick + 1e-6) - self.steps",
     "k = int(seconds / self.tick)", TC),
    ("control: мышь без торможения", C, "speed = np.minimum(v_max, np.sqrt(2.0 * a_max * gain * dist))",
     "speed = v_max + 0.0 * dist", TC),
]
