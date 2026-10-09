"""Measure how often each player picks a move that is optimal according to the exact solver.

Positions: random prefixes of N plies (N in [min_ply, max_ply]) from the empty board, fixed seed.
 - optimal:   the chosen move has the best exact score (also the fastest win / slowest loss).
 - preserving: the chosen move keeps the game-theoretic outcome (win stays win, draw stays draw).
   Computed only on positions that are not already lost (in a lost position every move "preserves"
   the loss, which would inflate the score).
"""
import sys, random, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.games.connect4 import Connect4
from src.players.minimax import MinimaxPlayer
from src.players.mcts import MCTSPlayer
from src.players.random_player import RandomPlayer
from src.solver_ref import solve_many, best_value, sign

g = Connect4()


def sample_positions(n, min_ply=6, max_ply=24, seed=123):
    rng, out = random.Random(seed), set()
    while len(out) < n:
        k, s, seq = rng.randint(min_ply, max_ply), g.initial_state(), []
        for _ in range(k):
            if g.result(s) is not None:
                break
            m = rng.choice(g.legal_moves(s)); seq.append(m); s = g.play(s, m)
        if g.result(s) is None and len(seq) == k:
            out.add(tuple(seq))
    return sorted(out)


def replay(seq):
    s = g.initial_state()
    for m in seq:
        s = g.play(s, m)
    return s


def score_player(player, positions, ref):
    opt = pres = n_live = 0
    for seq in positions:
        m = player.choose(g, replay(seq))
        sc = ref[seq]
        opt += sc[m] == best_value(sc)
        if best_value(sc) >= 0:
            n_live += 1
            pres += sign(sc[m]) == sign(best_value(sc))
    return opt / len(positions), pres / n_live


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    positions = sample_positions(n)
    t = time.time()
    ref = solve_many(positions)
    print(f"solved {n} positions in {time.time()-t:.0f}s", flush=True)
    wins = sum(best_value(v) > 0 for v in ref.values()); draws = sum(best_value(v) == 0 for v in ref.values())
    print(f"position values for player to move: {wins} won, {draws} drawn, {n-wins-draws} lost")
    players = [RandomPlayer(0)] + [MinimaxPlayer(d, seed=0) for d in (1, 2, 4, 6)] + \
              [MCTSPlayer(s, seed=0) for s in (100, 500, 2000)]
    for p in players:
        o, pr = score_player(p, positions, ref)
        print(f"{p.name:12s} optimal={o:.3f}  outcome-preserving (won/drawn positions)={pr:.3f}", flush=True)
