"""Play games between two players, alternating who starts, with a Wilson 95% CI."""
import math


def play_game(game, first, second, verbose=False):
    """Return +1 if `first` wins, -1 if `second` wins, 0 draw."""
    players, state, turn = (first, second), game.initial_state(), 0
    while game.result(state) is None:
        state = game.play(state, players[turn].choose(game, state))
        turn ^= 1
        if verbose:
            print(game.render(state), "\n")
    r = game.result(state)  # value for the player to move next, i.e. players[turn]
    return r if turn == 0 else -r


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def evaluate(game, a, b, n_games=200):
    """a vs b, alternating the starting player. Score of `a` as W/D/L + win-rate CI."""
    w = d = l = 0
    for i in range(n_games):
        r = play_game(game, a, b) if i % 2 == 0 else -play_game(game, b, a)
        w, d, l = w + (r == 1), d + (r == 0), l + (r == -1)
    lo, hi = wilson(w + d / 2, n_games)  # draws count half
    return {"wins": w, "draws": d, "losses": l, "score": (w + d / 2) / n_games, "ci95": (lo, hi)}
