"""AlphaZero-style training on Connect 4: self-play -> train -> evaluate vs exact solver, repeated.

Outputs (git-ignored): runs/<name>/ckpt_XXX.pt, runs/<name>/log.jsonl, runs/<name>/config.json
"""
import argparse, collections, json, pathlib, random, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import copy, math
import numpy as np
import torch
import torch.nn.functional as F

from src.games.registry import make_game
from src.alphazero.net import PolicyValueNet
from src.alphazero.selfplay import self_play
from src.alphazero.arena import arena
from src.alphazero.search import best_moves
from src.solver_ref import solve_many, sample_positions, replay, best_value, sign

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="connect4", choices=["connect4", "dobutsu"])
ap.add_argument("--name", default="run")
ap.add_argument("--dirichlet", type=float, default=1.0, help="Dirichlet noise alpha at the root in self-play")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--games", type=int, default=64, help="self-play games per iteration")
ap.add_argument("--sims", type=int, default=100, help="MCTS simulations per move in self-play")
ap.add_argument("--steps", type=int, default=150, help="gradient steps per iteration")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--buffer", type=int, default=100000, help="replay buffer size (positions)")
ap.add_argument("--channels", type=int, default=64)
ap.add_argument("--blocks", type=int, default=3)
ap.add_argument("--eval-sims", type=int, default=100)
ap.add_argument("--eval-positions", type=int, default=200)
ap.add_argument("--lr-final-frac", type=float, default=0.1, help="cosine decay of lr down to this fraction")
ap.add_argument("--arena-games", type=int, default=40, help="0 disables gating (candidate always accepted)")
ap.add_argument("--arena-sims", type=int, default=50)
ap.add_argument("--threshold", type=float, default=0.55, help="min arena score to replace the best net")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
out = pathlib.Path(__file__).resolve().parents[1] / "runs" / a.name
out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(vars(a), indent=1))

game, shape = make_game(a.game)
net = PolicyValueNet(**shape, channels=a.channels, blocks=a.blocks)
opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
best = copy.deepcopy(net).eval()   # network that generates self-play data (gated)
buffer = collections.deque(maxlen=a.buffer)
print(f"params: {sum(p.numel() for p in net.parameters()):,}", flush=True)

HAS_SOLVER = a.game == "connect4"
if HAS_SOLVER:
    positions = sample_positions(game, a.eval_positions)
    ref = solve_many(positions)  # cached exact-solver scores
    states = [replay(game, s) for s in positions]


def solver_metrics(sims):
    if not HAS_SOLVER:
        return float("nan"), float("nan")
    moves = best_moves(game, best, states, sims)
    opt_ = pres = live = 0
    for seq, m in zip(positions, moves):
        sc = ref[seq]
        opt_ += sc[m] == best_value(sc)
        if best_value(sc) >= 0:
            live += 1
            pres += sign(sc[m]) == sign(best_value(sc))
    return opt_ / len(positions), pres / live


def train_steps(n):
    net.train()
    data = list(buffer)
    tot_p = tot_v = 0.0
    for _ in range(n):
        idx = np.random.randint(len(data), size=a.batch)
        x = torch.from_numpy(np.stack([data[i][0] for i in idx]))
        pi = torch.from_numpy(np.stack([data[i][1] for i in idx]))
        z = torch.tensor([data[i][2] for i in idx])
        logits, v = net(x)
        lp = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean()   # cross-entropy to MCTS visits
        lv = F.mse_loss(v, z)
        opt.zero_grad(); (lp + lv).backward(); opt.step()
        tot_p += lp.item(); tot_v += lv.item()
    net.eval()
    return tot_p / n, tot_v / n


def save(i):
    torch.save({"config": best.config, "model": best.state_dict(), "iter": i}, out / f"ckpt_{i:03d}.pt")


save(0)
o, p = solver_metrics(0)
print(f"iter 0 (untrained): net-only optimal={o:.3f}", flush=True)
t_start = time.time()
for it in range(1, a.iters + 1):
    t0 = time.time()
    frac = (it - 1) / max(a.iters - 1, 1)
    lr = a.lr * (a.lr_final_frac + (1 - a.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
    for grp in opt.param_groups:
        grp['lr'] = lr
    examples, stats = self_play(game, best, a.games, a.sims, rng, noise=(0.25, a.dirichlet))
    t_sp = time.time() - t0
    for planes, pi, z in examples:                      # augment with the mirror image
        for pl, p_ in game.symmetries(planes, pi):
            buffer.append((pl, p_, z))
    t1 = time.time()
    lp, lv = train_steps(a.steps)
    t_tr = time.time() - t1
    t2 = time.time()
    if a.arena_games:
        arena_score = arena(game, net, best, a.arena_games, a.arena_sims, rng)
        promoted = arena_score >= a.threshold
    else:
        arena_score, promoted = None, True
    if promoted:
        best.load_state_dict(net.state_dict())
    else:
        net.load_state_dict(best.state_dict())   # discard a candidate that is not better
    t_ar = time.time() - t2
    o_net, p_net = solver_metrics(0)
    o_srch, p_srch = solver_metrics(a.eval_sims)
    save(it)
    rec = dict(iter=it, lr=lr, arena_score=arena_score, promoted=promoted, arena_s=t_ar, loss_policy=lp, loss_value=lv, buffer=len(buffer), selfplay_s=t_sp, train_s=t_tr,
               total_s=time.time() - t_start, solver_optimal_net=o_net, solver_optimal_search=o_srch,
               solver_preserving_search=p_srch, **{k: v for k, v in stats.items() if k != "lengths"})
    (out / "log.jsonl").open("a").write(json.dumps(rec) + "\n")
    print(f"it {it:3d} | loss p={lp:.3f} v={lv:.3f} | selfplay {t_sp:.0f}s train {t_tr:.0f}s | len={stats['avg_len']:.1f} "
          f"1st-win={stats['first_player_win_rate']:.2f} draw={stats['draw_rate']:.2f} | arena={arena_score} {'PROMOTED' if promoted else 'rejected'} | "
          f"solver optimal: net={o_net:.3f} search{a.eval_sims}={o_srch:.3f} (outcome kept {p_srch:.3f})", flush=True)
