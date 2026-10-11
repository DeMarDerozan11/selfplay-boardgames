"""Ablation: train the same network from the games of a fixed opponent instead of self-play.

Data: `--games` games of minimax (depth `--depth`) against itself, 15% of the moves replaced by a random
move for diversity. Policy target = the move minimax would play (one-hot), value target = game outcome.
Same network, optimiser, number of gradient steps and learning-rate schedule as scripts/train.py.
Writes runs/<name>/ckpt_<iters>.pt so that scripts/eval_curve.py can evaluate it.
"""
import argparse, json, math, pathlib, random, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from src.games.registry import make_game
from src.alphazero.net import PolicyValueNet
from src.players.minimax import MinimaxPlayer

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="connect4")
ap.add_argument("--name", required=True)
ap.add_argument("--games", type=int, default=2500)
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--iters", type=int, default=20, help="only used to name the checkpoint")
ap.add_argument("--steps", type=int, default=3000, help="total gradient steps")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--lr-final-frac", type=float, default=0.1)
ap.add_argument("--channels", type=int, default=64)
ap.add_argument("--blocks", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
out = pathlib.Path(__file__).resolve().parents[1] / "runs" / a.name
out.mkdir(parents=True, exist_ok=True)
game, shape = make_game(a.game)
rng = random.Random(a.seed)
player = MinimaxPlayer(a.depth, seed=a.seed)

t0 = time.time()
data = []
for _ in range(a.games):
    s, hist = game.initial_state(), []
    while game.result(s) is None:
        best = player.choose(game, s)
        pi = np.zeros(game.num_actions, np.float32); pi[best] = 1.0
        hist.append((game.encode(s), pi))
        move = rng.choice(game.legal_moves(s)) if rng.random() < 0.15 else best
        s = game.play(s, move)
    r, T = game.result(s), len(hist)
    for t, (planes, pi) in enumerate(hist):
        z = r if (T - t) % 2 == 0 else -r
        for pl, p_ in game.symmetries(planes, pi):
            data.append((pl, p_, np.float32(z)))
print(f"generated {len(data)} positions from {a.games} games in {time.time()-t0:.0f}s", flush=True)

net = PolicyValueNet(**shape, channels=a.channels, blocks=a.blocks)
opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
for step in range(a.steps):
    frac = step / max(a.steps - 1, 1)
    for grp in opt.param_groups:
        grp["lr"] = a.lr * (a.lr_final_frac + (1 - a.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
    idx = np.random.randint(len(data), size=a.batch)
    x = torch.from_numpy(np.stack([data[i][0] for i in idx]))
    pi = torch.from_numpy(np.stack([data[i][1] for i in idx]))
    z = torch.tensor([data[i][2] for i in idx])
    logits, v = net(x)
    loss = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean() + F.mse_loss(v, z)
    opt.zero_grad(); loss.backward(); opt.step()
net.eval()
torch.save({"config": net.config, "model": net.state_dict(), "iter": a.iters}, out / f"ckpt_{a.iters:03d}.pt")
(out / "config.json").write_text(json.dumps(vars(a), indent=1))
print(f"done in {time.time()-t0:.0f}s, final loss {loss.item():.3f}")
