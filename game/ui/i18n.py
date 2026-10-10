"""Строки окна на русском и английском."""
from aikit.i18n import Translator

RU = {
    "app_title": "Пинг-понг и аэрохоккей",
    "app_sub": "против ИИ, который научился играть сам: реагирует с задержкой и не быстрее вас",
    "game": "Игра", "hockey": "Аэрохоккей", "pong": "Пинг-понг",
    "level": "Уровень", "novice": "Новичок", "medium": "Средний", "strong": "Сильный",
    "mode": "Режим", "mode_ai": "Против ИИ", "mode_duo": "Вдвоём", "mode_demo": "ИИ против ИИ",
    "play": "Играть", "stats": "Статистика", "settings": "Настройки", "menu": "В меню", "again": "Заново",
    "pause": "Пауза", "resume": "Продолжить",
    "you": "Вы", "ai": "ИИ", "player2": "Игрок 2",
    "thoughts": "Мысли ИИ",
    "thoughts_hint": "Пунктир - куда, по расчёту ИИ, полетит шайба или мяч. Кружок - куда ИИ идёт бить. "
                     "ИИ видит прошлое (задержка реакции) и не видит вращения мяча, поэтому иногда ошибается.",
    "controls_hockey": "Мышь - бита (в своей половине). Пробел - пауза, Esc - меню.",
    "controls_pong": "Мышь - ракетка, колесо - наклон. Пробел - пауза, Esc - меню.",
    "controls_duo_hockey": "Игрок 2 справа: стрелки.",
    "controls_duo_pong": "Игрок 2 справа: стрелки - ракетка, PageUp и PageDown - наклон.",
    "score_to": "до {n}",
    "serve_you": "Подаёте вы: ракетку под мяч и вперёд", "serve_ai": "Подаёт ИИ", "serve_p2": "Подаёт игрок 2",
    "serve_left": "Подаёт левый", "serve_right": "Подаёт правый",
    "win": "Вы выиграли!", "loss": "Победил ИИ", "left_won": "Победил левый", "right_won": "Победил правый",
    "paused": "Пауза",
    "no_net": "Модель ещё не обучена: вместо сети играет бот-«предсказатель» с задержкой уровня.",
    "stats_title": "Статистика", "your_games": "Ваши партии против ИИ",
    "col_setup": "Игра и уровень", "col_games": "Партий", "col_wins": "Побед", "col_losses": "Поражений",
    "col_streak": "Лучшая серия",
    "passport": "Паспорт ИИ", "not_trained": "Модель ещё не обучена.",
    "passport_text": "Сеть: {features} входов, слои {hidden}, {params} весов. Обучение: подражание боту "
                     "({imitation} примеров), затем {gens} поколений CEM в игре с собой и лигой, "
                     "{games} партий, {time}.",
    "col_check": "Проверка", "col_result": "Доля побед", "col_points": "Очки",
    "checks_text": "{n} партий до конца на каждую проверку. Боты играют с задержкой 0.125 с.",
    "vs": "против", "follower": "«следящий»", "predictor": "«предсказатель»", "nohint": "без подсказки",
    "settings_title": "Настройки",
    "theme": "Тема", "theme_hint": "Тёмная или светлая, как в Paint.", "dark": "Тёмная", "light": "Светлая",
    "language": "Язык", "language_hint": "Русский или английский.",
    "volume": "Громкость", "volume_hint": "Звук ударов и голов.",
    "strict_serve": "Подача по правилам",
    "strict_serve_hint": "Мяч сначала отскакивает на вашей стороне, потом у соперника. ИИ обучен на простой "
                         "подаче (сразу к сопернику), поэтому правило действует только для людей.",
    "saved_note": "Настройки сохраняются сразу.",
    "seconds": "{n} с", "minutes": "{n} мин", "hours": "{n} ч",
    "footer": "PinPongAi {v} · физика 240 шагов в секунду",
    "goal": "Гол!", "point": "Очко",
}

EN = {
    "app_title": "Ping-pong & Air Hockey",
    "app_sub": "against a self-taught AI that reacts with a delay and is no faster than you",
    "game": "Game", "hockey": "Air hockey", "pong": "Ping-pong",
    "level": "Level", "novice": "Novice", "medium": "Medium", "strong": "Strong",
    "mode": "Mode", "mode_ai": "Versus AI", "mode_duo": "Two players", "mode_demo": "AI vs AI",
    "play": "Play", "stats": "Statistics", "settings": "Settings", "menu": "Menu", "again": "Restart",
    "pause": "Pause", "resume": "Resume",
    "you": "You", "ai": "AI", "player2": "Player 2",
    "thoughts": "AI thoughts",
    "thoughts_hint": "Dashed line - where the AI expects the puck or ball to go. Circle - where it moves to hit. "
                     "The AI sees the past (reaction delay) and cannot see spin, so it sometimes misjudges.",
    "controls_hockey": "Mouse - mallet (in your half). Space - pause, Esc - menu.",
    "controls_pong": "Mouse - paddle, wheel - tilt. Space - pause, Esc - menu.",
    "controls_duo_hockey": "Player 2 on the right: arrow keys.",
    "controls_duo_pong": "Player 2 on the right: arrows move, PageUp and PageDown tilt.",
    "score_to": "to {n}",
    "serve_you": "Your serve: paddle under the ball and forward", "serve_ai": "AI serves",
    "serve_p2": "Player 2 serves", "serve_left": "Left serves", "serve_right": "Right serves",
    "win": "You won!", "loss": "The AI won", "left_won": "Left player won", "right_won": "Right player won",
    "paused": "Paused",
    "no_net": "The model is not trained yet: a predictor bot with the level's delay plays instead.",
    "stats_title": "Statistics", "your_games": "Your games against the AI",
    "col_setup": "Game and level", "col_games": "Games", "col_wins": "Wins", "col_losses": "Losses",
    "col_streak": "Best streak",
    "passport": "AI passport", "not_trained": "The model is not trained yet.",
    "passport_text": "Network: {features} inputs, layers {hidden}, {params} weights. Training: imitating a bot "
                     "({imitation} examples), then {gens} CEM generations in self-play with a league, "
                     "{games} games, {time}.",
    "col_check": "Check", "col_result": "Win rate", "col_points": "Points",
    "checks_text": "{n} full games per check. Bots play with a 0.125 s delay.",
    "vs": "vs", "follower": "follower", "predictor": "predictor", "nohint": "no hint",
    "settings_title": "Settings",
    "theme": "Theme", "theme_hint": "Dark or light, as in Paint.", "dark": "Dark", "light": "Light",
    "language": "Language", "language_hint": "Russian or English.",
    "volume": "Volume", "volume_hint": "Sound of hits and goals.",
    "strict_serve": "Serve by the rules",
    "strict_serve_hint": "The ball bounces on your side first, then on the opponent's. The AI was trained on the "
                         "simple serve (straight to the opponent), so the rule applies to people only.",
    "saved_note": "Settings are saved at once.",
    "seconds": "{n} s", "minutes": "{n} min", "hours": "{n} h",
    "footer": "PinPongAi {v} · physics at 240 steps per second",
    "goal": "Goal!", "point": "Point",
}


def translator(lang):
    return Translator({"ru": RU, "en": EN}, lang)
