"""Batched head-to-head between two networks (used to gate which network generates self-play)."""
import numpy as np

from src.alphazero.search import Tree, run_searches, visit_counts


def arena(game, net_a, net_b, n_games, sims, rng, sample_plies=4):
    """Play `n_games` between a and b, alternating who starts. The first `sample_plies` moves are
    sampled from visit counts (otherwise games would repeat); then greedy. Returns a's score in [0,1]
    (win = 1, draw = 0.5)."""
    states = [game.initial_state() for _ in range(n_games)]
    a_first = [i % 2 == 0 for i in range(n_games)]
    score = [None] * n_games
    active, ply = list(range(n_games)), 0
    while active:
        for use_a in (True, False):
            idx = [i for i in active if (a_first[i] == (ply % 2 == 0)) == use_a]
            if not idx:
                continue
            trees = [Tree(game, states[i]) for i in idx]
            run_searches(game, net_a if use_a else net_b, trees, sims)
            for i, t in zip(idx, trees):
                counts = visit_counts(t)
                m = int(rng.choice(len(counts), p=counts / counts.sum())) if ply < sample_plies \
                    else int(np.argmax(counts))
                states[i] = game.play(states[i], m)
                r = game.result(states[i])
                if r is not None:  # r = value for the player now to move; mover just played
                    mover_is_a = use_a
                    score[i] = 0.5 if r == 0 else (1.0 if mover_is_a else 0.0)  # r == -1: mover won
        active = [i for i in active if score[i] is None]
        ply += 1
    return float(np.mean(score))
