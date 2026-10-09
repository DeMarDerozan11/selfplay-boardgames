"""Minimax (various depths) vs random, alternating starts, with timing."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.games.connect4 import Connect4
from src.players.minimax import MinimaxPlayer
from src.players.random_player import RandomPlayer
from src.engine.match import evaluate

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
g = Connect4()
for d in (1, 2, 4, 6):
    mm = MinimaxPlayer(depth=d, seed=0)
    t = time.time()
    r = evaluate(g, mm, RandomPlayer(0), n)
    print(f"depth {d}: {r['wins']}W {r['draws']}D {r['losses']}L  score={r['score']:.3f} "
          f"CI95=({r['ci95'][0]:.3f},{r['ci95'][1]:.3f})  {time.time()-t:.1f}s for {n} games", flush=True)
