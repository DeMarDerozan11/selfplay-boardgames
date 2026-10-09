"""Strength of AlphaZero checkpoints in matches vs fixed references (alternating starts, fixed seeds)."""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src.games.connect4 import Connect4
from src.alphazero.net import PolicyValueNet
from src.alphazero.search import AZPlayer
from src.players.minimax import MinimaxPlayer
from src.players.mcts import MCTSPlayer
from src.players.random_player import RandomPlayer
from src.engine.match import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="main")
ap.add_argument("--iters", type=int, nargs="+", default=[0, 10, 20, 40])
ap.add_argument("--games", type=int, default=40)
ap.add_argument("--sims", type=int, default=100)
a = ap.parse_args()
torch.set_num_threads(4)
g = Connect4()
run = pathlib.Path(__file__).resolve().parents[1] / "runs" / a.run
out = run / "match_results.jsonl"
for it in a.iters:
    net = PolicyValueNet.from_checkpoint(run / f"ckpt_{it:03d}.pt")
    for opp in (RandomPlayer(0), MinimaxPlayer(4, seed=0), MCTSPlayer(500, seed=0)):
        t = time.time()
        r = evaluate(g, AZPlayer(net, a.sims), opp, a.games)
        rec = dict(run=a.run, iter=it, sims=a.sims, opponent=opp.name, games=a.games, **{k: r[k] for k in ("wins", "draws", "losses", "score")}, ci95=r["ci95"])
        out.open("a").write(json.dumps(rec) + "\n")
        print(f"iter {it:2d} az-{a.sims} vs {opp.name:12s}: {r['wins']}W {r['draws']}D {r['losses']}L score={r['score']:.3f} "
              f"CI=({r['ci95'][0]:.2f},{r['ci95'][1]:.2f}) {time.time()-t:.0f}s", flush=True)
