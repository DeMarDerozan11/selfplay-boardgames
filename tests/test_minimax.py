import random
from src.games.connect4 import Connect4
from src.players.minimax import MinimaxPlayer, WIN

g = Connect4()


def play_seq(cols):
    s = g.initial_state()
    for c in cols:
        s = g.play(s, c)
    return s


def plain_negamax(state, depth):
    """Reference: no pruning, no table, no ordering."""
    r = g.result(state)
    if r is not None:
        return r * (WIN + depth)
    if depth == 0:
        return g.heuristic(state)
    return max(-plain_negamax(g.play(state, m), depth - 1) for m in g.legal_moves(state))


def test_matches_plain_minimax_on_random_positions():
    rng = random.Random(0)
    p = MinimaxPlayer(depth=4)
    for _ in range(15):
        s = g.initial_state()
        for _ in range(rng.randint(0, 14)):
            if g.result(s) is not None:
                break
            s = g.play(s, rng.choice(g.legal_moves(s)))
        if g.result(s) is not None:
            continue
        got = p.move_scores(g, s)
        want = {m: -plain_negamax(g.play(s, m), 3) for m in g.legal_moves(s)}
        assert got == want  # alpha-beta + TT must not change any root move value


def test_takes_immediate_win():
    s = play_seq([0, 1, 0, 1, 0, 1])  # X to move, can win in column 0
    assert MinimaxPlayer(depth=2).choose(g, s) == 0


def test_blocks_immediate_loss():
    s = play_seq([0, 1, 0, 1, 0])  # X has 3 in col 0, O to move must block
    assert MinimaxPlayer(depth=2).choose(g, s) == 0


def test_pruning_reduces_nodes():
    s = play_seq([3, 3, 2])
    a, b = MinimaxPlayer(depth=5), MinimaxPlayer(depth=5, use_tt=False, use_ordering=False)
    a.move_scores(g, s); b.move_scores(g, s)
    assert a.nodes < b.nodes
