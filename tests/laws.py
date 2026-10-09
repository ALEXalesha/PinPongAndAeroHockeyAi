"""Сколько примеров давать законам, которые гоняют физику шаг за шагом: такой пример в
сотни раз дороже чистой функции. Обычный прогон - SIM примеров, HYPOTHESIS_PROFILE=thorough -
в 10 раз больше."""
import os

from hypothesis import settings

SCALE = 10 if os.environ.get("HYPOTHESIS_PROFILE") == "thorough" else 1


def sim(n=60):
    return settings(max_examples=n * SCALE, deadline=None)
