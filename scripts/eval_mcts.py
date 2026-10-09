"""MCTS vs random and vs minimax, alternating starts."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.games.connect4 import Connect4
from src.players.minimax import MinimaxPlayer
from src.players.mcts import MCTSPlayer
from src.players.random_player import RandomPlayer
from src.engine.match import evaluate

n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
g = Connect4()

def run(a, b):
    t = time.time()
    r = evaluate(g, a, b, n)
    print(f"{a.name} vs {b.name}: {r['wins']}W {r['draws']}D {r['losses']}L score={r['score']:.3f} "
          f"CI95=({r['ci95'][0]:.3f},{r['ci95'][1]:.3f}) {time.time()-t:.0f}s/{n} games", flush=True)

for sims in (100, 500, 2000):
    run(MCTSPlayer(sims, seed=0), RandomPlayer(0))
for sims in (100, 500, 2000):
    run(MCTSPlayer(sims, seed=0), MinimaxPlayer(4, seed=0))
