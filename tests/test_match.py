from src.games.connect4 import Connect4
from src.engine.match import play_game, evaluate


class Scripted:
    def __init__(self, cols):
        self.cols = iter(cols)

    def choose(self, game, state):
        return next(self.cols)


def test_play_game_sign_convention():
    g = Connect4()
    # first player stacks column 0 and wins; second wastes moves in column 1
    assert play_game(g, Scripted([0, 0, 0, 0]), Scripted([1, 1, 1])) == 1
    # second player stacks column 1 and wins
    assert play_game(g, Scripted([0, 0, 0, 2]), Scripted([1, 1, 1, 1])) == -1


def test_evaluate_alternates_starter():
    g = Connect4()
    always0 = type("A", (), {"choose": lambda self, g, s: g.legal_moves(s)[0]})()
    r = evaluate(g, always0, always0, 10)
    # identical deterministic players: the first mover wins every game, so a's score is exactly 50%
    assert r["wins"] == 5 and r["losses"] == 5
