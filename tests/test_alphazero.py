import numpy as np
import torch

from src.games.connect4 import Connect4
from src.alphazero.net import PolicyValueNet
from src.alphazero.search import Tree, run_searches, visit_counts, best_moves
from src.alphazero.selfplay import self_play

g = Connect4()


class Uniform(torch.nn.Module):
    """Fake network: uniform policy, zero value. The search must then rely on real outcomes."""
    def forward(self, x):
        return torch.zeros(len(x), 7), torch.zeros(len(x))


def play_seq(cols):
    s = g.initial_state()
    for c in cols:
        s = g.play(s, c)
    return s


def test_encode_planes():
    s = play_seq([3, 4])  # X in col 3, O in col 4; X to move
    p = g.encode(s)
    assert p.shape == (2, 6, 7) and p[0, 0, 3] == 1 and p[1, 0, 4] == 1 and p.sum() == 2
    s = play_seq([3])  # O to move: planes swap roles
    assert g.encode(s)[1, 0, 3] == 1 and g.encode(s)[0].sum() == 0


def test_mirror_symmetry():
    s = play_seq([0, 1, 2])
    mirrored = play_seq([6, 5, 4])
    assert np.array_equal(g.symmetries(g.encode(s), np.arange(7.0))[1][0], g.encode(mirrored))


def test_visits_sum_to_sims_and_legal_only():
    s = play_seq([0] * 6)  # column 0 is full
    t = Tree(g, s)
    run_searches(g, Uniform(), [t], 60)
    n = visit_counts(t)
    assert n.sum() == 60 and n[0] == 0


def test_finds_immediate_win_and_block():
    assert best_moves(g, Uniform(), [play_seq([0, 1, 0, 1, 0, 1])], 200) == [0]  # win
    assert best_moves(g, Uniform(), [play_seq([0, 1, 0, 1, 0])], 400) == [0]     # block


def test_self_play_labels_consistent():
    net = PolicyValueNet(2, 6, 7, 7, channels=8, blocks=1)
    ex, stats = self_play(g, net, 8, 10, np.random.default_rng(0))
    assert len(ex) == sum(stats["lengths"])
    i = 0
    for T in stats["lengths"]:
        zs = [float(e[2]) for e in ex[i:i + T]]
        i += T
        if zs[-1] == 0:                      # draw
            assert all(z == 0 for z in zs)
        else:                                # the player who made the last move won
            assert zs[-1] == 1 and all(zs[k] == -zs[k + 1] for k in range(T - 1))
    for planes, pi, z in ex:
        assert abs(pi.sum() - 1) < 1e-5
