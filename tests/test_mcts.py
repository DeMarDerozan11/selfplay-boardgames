from src.games.connect4 import Connect4
from src.players.mcts import MCTSPlayer

g = Connect4()


def play_seq(cols):
    s = g.initial_state()
    for c in cols:
        s = g.play(s, c)
    return s


def test_takes_immediate_win():
    s = play_seq([0, 1, 0, 1, 0, 1])  # X wins by playing column 0
    assert MCTSPlayer(300, seed=0).choose(g, s) == 0


def test_blocks_immediate_loss():
    s = play_seq([0, 1, 0, 1, 0])  # O must block column 0
    assert MCTSPlayer(1000, seed=0).choose(g, s) == 0


def test_visits_sum_to_simulations():
    visits = MCTSPlayer(200, seed=0).search(g, g.initial_state())
    assert sum(visits.values()) == 200


def test_seed_reproducible():
    s = play_seq([3, 3])
    assert MCTSPlayer(100, seed=5).search(g, s) == MCTSPlayer(100, seed=5).search(g, s)
