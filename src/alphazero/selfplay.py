"""Self-play data generation: many games in parallel, one search tree per game."""
import numpy as np

from src.alphazero.search import Tree, run_searches, visit_counts


def self_play(game, net, n_games, sims, rng, c_puct=1.5, noise=(0.25, 1.0), temp_moves=8):
    """Play `n_games` games against itself. Returns (examples, stats).

    example = (planes, pi, z): the network input, the MCTS visit distribution (training target
    for the policy) and the final outcome from the point of view of the player to move (z).
    The first `temp_moves` moves are sampled from the visit distribution (diversity), the rest
    are greedy.
    """
    states = [game.initial_state() for _ in range(n_games)]
    history = [[] for _ in range(n_games)]  # (planes, pi) per ply
    active = list(range(n_games))
    results = [None] * n_games
    ply = 0
    while active:
        trees = [Tree(game, states[i], c_puct, noise, rng) for i in active]
        run_searches(game, net, trees, sims)
        still = []
        for i, t in zip(active, trees):
            counts = visit_counts(t)
            pi = counts / counts.sum()
            history[i].append((game.encode(states[i]), pi.astype(np.float32)))
            if ply < temp_moves:
                move = int(rng.choice(len(pi), p=pi))
            else:
                move = int(np.argmax(counts))
            states[i] = game.play(states[i], move)
            r = game.result(states[i])
            if r is None:
                still.append(i)
            else:
                results[i] = r  # value for the player to move in the final state
        active = still
        ply += 1

    examples, lengths, first_player_wins, draws = [], [], 0, 0
    for i in range(n_games):
        T, r = len(history[i]), results[i]  # final state is at ply T, its player to move has value r
        for t, (planes, pi) in enumerate(history[i]):
            z = r if (T - t) % 2 == 0 else -r
            examples.append((planes, pi, np.float32(z)))
        lengths.append(T)
        draws += r == 0
        first_player_wins += (r == -1 and T % 2 == 1) or (r == 1 and T % 2 == 0)
    stats = dict(games=n_games, lengths=lengths, avg_len=float(np.mean(lengths)), draw_rate=draws / n_games,
                 first_player_win_rate=first_player_wins / n_games)
    return examples, stats
