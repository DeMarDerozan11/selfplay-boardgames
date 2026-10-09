"""Wrapper around the external exact Connect 4 solver (Pascal Pons, AGPL-3.0).

Positions are given as move sequences (our columns 0..6). The solver is run as a separate
process and results are cached on disk, because solving shallow positions without the
opening book is slow (seconds to minutes).
"""
import json
import pathlib
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
