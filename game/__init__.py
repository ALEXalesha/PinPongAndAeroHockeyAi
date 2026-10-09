"""PinPongAi: пинг-понг и аэрохоккей против ИИ, который научился играть сам.

Вход игры по контракту PythonAiGames: GAME_INFO и create_widget(ctx). Qt загружается
только в create_widget - физика, ИИ и обучение работают без него.
"""
from game.version import VERSION

GAME_INFO = {
    "id": "pinpongai",
    "title": {"ru": "Пинг-понг и аэрохоккей", "en": "Ping-pong & Air Hockey"},
    "summary": {"ru": "Пинг-понг и аэрохоккей против ИИ с честной реакцией, обученного игрой с собой",
                "en": "Ping-pong and air hockey against a self-taught AI with honest reaction time"},
    "preview": "docs/preview.png",
    "version": VERSION,
}

APP_NAME = "PinPongAi"


def create_widget(ctx):
    """ctx: aikit.Context. Виджет с меню, полем и панелями."""
    from game.ui.widget import GameWidget
    return GameWidget(ctx)
