# PinPongAi - ping-pong and air hockey against a self-taught AI

Work in progress, version 0.1.0. Russian version: [README.ru.md](README.ru.md).

Two games in one program: ping-pong seen from the side and air hockey seen from above. The
opponent is a small neural network trained by playing against itself. It is honest: its
paddle has the same speed and acceleration limits as yours, it reacts with a delay that
depends on the level, and it sees only the position and velocity of the ball.

## Why the physics is written from scratch

Both games run on one 2D physics engine with a fixed 1/240 s step. The window can draw at
30 or 144 frames per second, the game still makes 240 steps per second, so a match with the
same inputs plays out the same way. The engine works on numpy arrays of many matches at once:
the window uses one match, training uses hundreds, and it is the same code.

Collisions are continuous. A naive engine moves the puck and then checks for overlap, so a
fast puck can jump over a thin wall in one step. Here the engine computes the moment inside
the step when the distance to the wall becomes equal to the radius. Tests fire the puck at
up to 5000 units per second (the rink is 2 units long) and it never leaves the rink or
passes through a mallet.

| Check | What it proves |
|---|---|
| elastic hit, no friction | speed relative to the surface is preserved |
| hit with friction | energy never grows |
| wall reflection | angle in equals angle out |
| batch vs single match | identical, bit for bit |
| 30 vs 144 frames per second | same number of steps, same match |

## Status

- done: physics of both games, rules (goals, the 7-second rule, serve rotation, 11 points
  with a two-point lead), law tests;
- next: AI opponents and training, the window.

## Running

    python -m venv .venv
    .venv\Scripts\pip install -r requirements-dev.txt
    .venv\Scripts\python -m pytest -q
    .venv\Scripts\python main.py --selftest result.txt

License: MIT.
