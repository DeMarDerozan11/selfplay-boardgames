"""Run the Connect 4 ablation grid (resumable): every configuration x seed is trained, then evaluated
in matches against minimax (depth 4) and MCTS (500 simulations) with 60 games each.

Common protocol (shorter than the main run so that the whole grid fits in a few hours):
20 iterations x 64 self-play games, 150 gradient steps per iteration, arena gating, cosine lr decay.
"""
import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
COMMON = ["--game", "connect4", "--iters", "20", "--games", "64", "--steps", "150", "--eval-positions", "40"]
CONFIGS = {  # name -> extra train.py arguments (None = special script)
    "base_sims100_net64x3": ["--sims", "100", "--channels", "64", "--blocks", "3"],
    "sims25": ["--sims", "25", "--channels", "64", "--blocks", "3"],
    "sims300": ["--sims", "300", "--channels", "64", "--blocks", "3"],
    "net16x1": ["--sims", "100", "--channels", "16", "--blocks", "1"],
    "net32x2": ["--sims", "100", "--channels", "32", "--blocks", "2"],
    "opponent_data": None,
}
SEEDS = [0, 1, 2]
only = sys.argv[1:]  # optionally restrict to some configuration names

for seed in SEEDS:
    for cfg, extra in CONFIGS.items():
        if only and cfg not in only:
            continue
        name = f"abl_{cfg}_s{seed}"
        run = ROOT / "runs" / name
        if not (run / "ckpt_020.pt").exists():
            if extra is None:
                cmd = [PY, "scripts/train_from_opponent.py", "--game", "connect4", "--name", name, "--seed", str(seed),
                       "--iters", "20", "--games", "2500", "--steps", "3000"]
            else:
                cmd = [PY, "scripts/train.py", "--name", name, "--seed", str(seed), *COMMON, *extra]
            print("TRAIN", name, flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=open(ROOT / "runs" / f"{name}.out", "w"), stderr=subprocess.STDOUT)
        res = run / "match_results.jsonl"
        if not res.exists():
            print("EVAL ", name, flush=True)
            subprocess.run([PY, "scripts/eval_curve.py", "--game", "connect4", "--run", name, "--iters", "20", "--games", "60",
                            "--opponents", "minimax", "mcts"], cwd=ROOT, check=True,
                           stdout=open(ROOT / "runs" / f"{name}.eval.out", "w"), stderr=subprocess.STDOUT)
print("ALL DONE", flush=True)
