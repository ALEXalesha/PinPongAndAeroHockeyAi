"""Обучение и паспорт: python -m game.train ...

    python -m game.train run --game hockey --run runs/hockey --gens 120          # обучение (с места)
    python -m game.train run --game hockey --run runs/hockey-nohint --nohint     # сеть без подсказки
    python -m game.train run --game pong --run runs/pong --probe                 # 2 поколения + прогноз
    python -m game.train export --game hockey --run runs/hockey --nohint-run runs/hockey-nohint
    python -m game.train passport --game hockey --games 200

Потоки BLAS ограничены до импорта numpy (--threads, по умолчанию 1).
"""
import argparse
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m game.train")
    ap.add_argument("cmd", choices=("run", "export", "passport"))
    ap.add_argument("--game", choices=("hockey", "pong"), required=True)
    ap.add_argument("--run", default=None, help="папка обучения: журнал и снимок")
    ap.add_argument("--nohint-run", default=None)
    ap.add_argument("--nohint", action="store_true", help="сеть без подсказки траектории")
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="2 поколения и прогноз времени")
    ap.add_argument("--games", type=int, default=200, help="партий на проверку паспорта")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    from aikit.trainlog import limit_threads
    limit_threads(args.threads, force=True)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "export":
        from game.train.export import export
        export(args.game, args.run, args.nohint_run)
        return 0
    from game.train.passport import make_passport
    make_passport(args.game, args.games, seed=args.seed)
    return 0


def run(args):
    import numpy as np

    from aikit.trainlog import TrainLog, load_snapshot, save_snapshot
    from game.train.evolve import DEFAULTS, State, generation

    run_dir = args.run or os.path.join("runs", args.game + ("-nohint" if args.nohint else ""))
    os.makedirs(run_dir, exist_ok=True)
    log = TrainLog(os.path.join(run_dir, "train.log"))
    snap = os.path.join(run_dir, "state.npz")
    loaded = None if args.probe else load_snapshot(snap)
    if loaded:
        st = State.restore(*loaded)
        log(f"продолжение с поколения {st.gen}")
    else:
        cfg = dict(DEFAULTS[args.game])
        st = State(args.game, cfg, args.seed, not args.nohint)
        from game.train.imitate import collect, fit
        t0 = time.time()
        X, Y = collect(args.game, n=64, seconds=8.0 if args.probe else 40.0, seed=args.seed, hint=st.hint)
        log(f"подражание: {len(X)} примеров собрано за {time.time() - t0:.0f} с")
        st.mean = fit(args.game, X, Y, st.cfg["hidden"], epochs=6 if args.probe else 40, seed=args.seed, log=log)
        st.league = [st.mean.copy()]
        st.snapshots[0] = st.mean.copy()
        st.imitation = len(X)
        st.seconds += time.time() - t0
        log(f"подражание готово за {time.time() - t0:.0f} с")
    if args.gens:
        st.cfg["gens"] = args.gens
    rng = np.random.default_rng([st.seed, st.gen])
    target = 2 if args.probe else st.cfg["gens"]
    log(f"{args.game}{' без подсказки' if not st.hint else ''}: параметров {st.dim}, популяция {st.cfg['pop']}, "
        f"партий на члена {st.cfg['games']} по {st.cfg['seconds']} с, поколений {st.cfg['gens']}")
    done = 0
    while (st.gen < target) if not args.probe else (done < target):
        row = generation(st, rng)
        done += 1
        log(" ".join(f"{k}={v}" for k, v in row.items()))
        if not args.probe:
            save_snapshot(snap, st.arrays(), st.meta())
    if args.probe:
        per = st.seconds / done
        log(f"пробный прогон: {per:.1f} с на поколение, на {st.cfg['gens']} поколений - "
            f"{per * st.cfg['gens'] / 60:.0f} мин")
    else:
        log(f"готово: {st.gen} поколений, {st.games_played} партий, {st.seconds / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
