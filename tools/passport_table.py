"""Таблица паспорта для README из game/models/*.json.

    python tools/passport_table.py ru|en
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {
    "ru": {"strong": "сильный", "medium": "средний", "novice": "новичок", "nohint": "сильный без подсказки",
           "follower": "«следящий»", "predictor": "«предсказатель»", "hockey": "Аэрохоккей", "pong": "Пинг-понг",
           "head": "| Проверка | Победы | Ничьи | Поражения | Доля побед | Средний счёт |",
           "train": "Обучение: подражание {imitation} примеров, {gens} поколений CEM, {games} партий, {min} мин.",
           "vs": "против"},
    "en": {"strong": "strong", "medium": "medium", "novice": "novice", "nohint": "strong without hint",
           "follower": "follower", "predictor": "predictor", "hockey": "Air hockey", "pong": "Ping-pong",
           "head": "| Check | Wins | Draws | Losses | Win rate | Mean score |",
           "train": "Training: imitation of {imitation} examples, {gens} CEM generations, {games} games, {min} min.",
           "vs": "vs"},
}


def table(lang):
    n = NAMES[lang]
    out = []
    for game in ("hockey", "pong"):
        path = os.path.join(ROOT, "game", "models", f"{game}.json")
        if not os.path.exists(path):
            continue
        pp = json.load(open(path, encoding="utf-8"))
        tr = pp["training"]
        out.append(f"**{n[game]}**, {pp['games_per_check']} " + ("партий до конца на проверку." if lang == "ru"
                                                                 else "full games per check."))
        out.append("")
        out.append(n["head"])
        out.append("|---|---|---|---|---|---|")
        for c in pp["checks"]:
            out.append(f"| {n[c['subject']]} {n['vs']} {n[c['opponent']]} | {c['wins']} | {c['draws']} | "
                       f"{c['losses']} | {100 * c['win_rate']:.1f}% | {c['points_for']:.2f} : {c['points_against']:.2f} |")
        out.append("")
        out.append(n["train"].format(imitation=f"{tr.get('imitation', 0):,}".replace(",", " "), gens=tr["gens"],
                                     games=f"{tr['games']:,}".replace(",", " "), min=round(tr["seconds"] / 60)))
        out.append("")
    return "\n".join(out).strip()


if __name__ == "__main__":
    print(table(sys.argv[1] if len(sys.argv) > 1 else "ru"))
