"""Tiny web demo (standard library only): play Connect 4 against the trained AlphaZero network.

The server is stateless: the browser sends the list of moves played so far. It returns the board,
the game status, and for AI moves the MCTS visit distribution and the network's value estimate.
"""
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from src.alphazero.net import PolicyValueNet
from src.alphazero.search import Tree, run_searches, visit_counts
from src.games.connect4 import Connect4, W, H

HERE = pathlib.Path(__file__).resolve().parent
game = Connect4()
LOCK = threading.Lock()
NET = None


def replay(moves):
    s = game.initial_state()
    for m in moves:
        if m not in game.legal_moves(s):
            raise ValueError("illegal move sequence")
        s = game.play(s, m)
    return s


def winning_cells(board):
    for r in range(H):
        for c in range(W):
            p = board[r][c]
            if p == ".":
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + k * dr, c + k * dc) for k in range(4)]
                if all(0 <= y < H and 0 <= x < W and board[y][x] == p for y, x in cells):
                    return cells
    return []


def describe(moves):
    s = replay(moves)
    board = [[game.cell(s, c, r) for c in range(W)] for r in range(H - 1, -1, -1)]  # top row first
    res = game.result(s)
    to_move = "X" if len(moves) % 2 == 0 else "O"
    status = "playing" if res is None else ("draw" if res == 0 else ("O" if to_move == "X" else "X"))
    win = winning_cells(board[::-1]) if status in ("X", "O") else []
    win = [[H - 1 - r, c] for r, c in win]  # to top-first row indexing
    return dict(board=board, status=status, to_move=to_move, legal=game.legal_moves(s), win=win)


def ai_move(moves, sims):
    s = replay(moves)
    with LOCK:
        tree = Tree(game, s)
        run_searches(game, NET, [tree], sims)
    n = visit_counts(tree)
    root = tree.root
    value = float(root.W.sum() / max(root.N.sum(), 1))  # expected outcome for the player to move
    return int(np.argmax(n)), (n / n.sum()).tolist(), value


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            moves = [int(m) for m in body.get("moves", [])]
            if self.path == "/api/state":
                self._send(200, describe(moves))
            elif self.path == "/api/ai":
                sims = max(1, min(int(body.get("sims", 100)), 3000))
                if describe(moves)["status"] != "playing":
                    raise ValueError("game is over")
                move, visits, value = ai_move(moves, sims)
                self._send(200, dict(move=move, visits=visits, value=value))
            else:
                self._send(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError) as e:
            self._send(400, {"error": str(e)})

    def log_message(self, *args):
        pass


def main(checkpoint, port=8000):
    global NET
    NET = PolicyValueNet.from_checkpoint(checkpoint)
    torch.set_num_threads(2)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Connect 4 demo on http://127.0.0.1:{port}  (checkpoint: {checkpoint})")
    srv.serve_forever()
