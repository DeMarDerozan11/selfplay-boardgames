"""Wrapper around the external exact Connect 4 solver (Pascal Pons, AGPL-3.0).

Positions are given as move sequences (our columns 0..6). The solver is run as a separate
process and results are cached on disk, because solving shallow positions without the
opening book is slow (seconds to minutes).
"""
import json
import pathlib
import random
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / "external" / "connect4-solver" / "c4solver"
CACHE = ROOT / "data" / "solver_cache.json"
INVALID = -1000


def _key(seq):
    return "".join(str(c + 1) for c in seq)  # solver notation: columns 1..7


def solve_many(seqs):
    """Return {tuple(seq): [score per column 0..6]}; score > 0 win, 0 draw, < 0 loss for the
    player to move; |score| = how quickly (bigger = faster win / slower loss). Full columns = -1000."""
    CACHE.parent.mkdir(exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = sorted({_key(s) for s in seqs} - cache.keys())
    if todo:
        out = subprocess.run([str(BINARY), "-a"], input="\n".join(todo) + "\n",
                             capture_output=True, text=True, check=True).stdout.splitlines()
        assert len(out) == len(todo), "solver rejected some positions"
        for k, line in zip(todo, out):
            parts = line.split()
            assert parts[0] == k
            cache[k] = [int(x) for x in parts[1:]]
        CACHE.write_text(json.dumps(cache))
    return {tuple(s): cache[_key(s)] for s in seqs}


def best_value(scores):
    return max(s for s in scores if s != INVALID)


def sign(x):
    return (x > 0) - (x < 0)


def sample_positions(game, n, min_ply=6, max_ply=24, seed=123):
    """Random-prefix positions (as move tuples), fixed seed, only non-terminal ones."""
    rng, out = random.Random(seed), set()
    while len(out) < n:
        k, s, seq = rng.randint(min_ply, max_ply), game.initial_state(), []
        for _ in range(k):
            if game.result(s) is not None:
                break
            m = rng.choice(game.legal_moves(s)); seq.append(m); s = game.play(s, m)
        if game.result(s) is None and len(seq) == k:
            out.add(tuple(seq))
    return sorted(out)


def replay(game, seq):
    s = game.initial_state()
    for m in seq:
        s = game.play(s, m)
    return s
