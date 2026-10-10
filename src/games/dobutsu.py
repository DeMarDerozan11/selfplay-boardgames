"""Dobutsu Shogi (どうぶつしょうぎ), 3 columns x 4 rows. See README for the exact rule set.

Position key (u64, identical to tools/dobutsu_solver.cpp so both implementations can be compared):
  bits 4*i..4*i+3 : cell i = row*3 + col (row 0 = first player's home row), code = type + 8*owner
                    types: 1 chick, 2 giraffe, 3 elephant, 4 lion, 5 hen (promoted chick)
  bits 48..59     : hands, 2 bits each for (owner, type in chick/giraffe/elephant)
  bit 60          : side to move (0 = first player, who moves "up")
Game state = (key, history of keys since the start). The history is only used for the
threefold-repetition draw and a length cap.

Actions are expressed from the point of view of the player to move (board rotated by 180 degrees
for the second player), so one network handles both sides:
  0..95   : move, action = from_square * 8 + direction   (directions in DIRS, forward = +row)
  96..131 : drop,  action = 96 + piece * 12 + square      (piece 0 chick, 1 giraffe, 2 elephant)
"""
from functools import lru_cache

import numpy as np

from src.engine.game import Game

ROWS, COLS = 4, 3
MAX_PLIES = 200  # game declared a draw beyond this length (keeps self-play finite)
CHICK, GIRAFFE, ELEPHANT, LION, HEN = 1, 2, 3, 4, 5
DIRS = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]  # (d_row, d_col), forward = +row
STEPS = {  # allowed directions per type (for the first player; mirrored for the second)
    CHICK: [(1, 0)],
    GIRAFFE: [(1, 0), (-1, 0), (0, 1), (0, -1)],
    ELEPHANT: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    LION: DIRS,
    HEN: [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)],
}
NUM_ACTIONS = 96 + 36
PIECE_CHARS = {CHICK: "c", GIRAFFE: "g", ELEPHANT: "e", LION: "l", HEN: "h"}


def _far_row(owner):
    return 3 if owner == 0 else 0


def encode_key(board, hands, side):
    k = 0
    for i, c in enumerate(board):
        k |= c << (4 * i)
    for o in range(2):
        for t in range(3):
            k |= hands[o][t] << (48 + (o * 3 + t) * 2)
    return k | (side << 60)


def decode_key(k):
    board = [(k >> (4 * i)) & 15 for i in range(12)]
    hands = [[(k >> (48 + (o * 3 + t) * 2)) & 3 for t in range(3)] for o in range(2)]
    return board, hands, (k >> 60) & 1


def initial_key():
    b = [0] * 12
    b[0], b[1], b[2], b[4] = ELEPHANT, LION, GIRAFFE, CHICK
    b[11], b[10], b[9], b[7] = ELEPHANT | 8, LION | 8, GIRAFFE | 8, CHICK | 8
    return encode_key(b, [[0] * 3, [0] * 3], 0)


def _rel_sq(sq, side):  # square index as seen by the player `side`
    return sq if side == 0 else 11 - sq


def _can_capture(board, side, target):
    for sq, code in enumerate(board):
        if code and (code >> 3) == side:
            r, c = divmod(sq, 3)
            for dr, dc in STEPS[code & 7]:
                if side:
                    dr = -dr
                if (r + dr) * 3 + (c + dc) == target and 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
                    return True
    return False


@lru_cache(maxsize=500_000)
def terminal_code(key):
    """0 = not terminal, 1 = player to move has won, 2 = player to move has lost."""
    board, _, s = decode_key(key)
    own = opp = -1
    for i, c in enumerate(board):
        if c == (LION | (s << 3)):
            own = i
        elif c == (LION | ((1 - s) << 3)):
            opp = i
    if own < 0:
        return 2                                   # my lion was captured
    if own // 3 == _far_row(s):
        return 1                                   # my lion stands on the far row and survived
    if opp < 0:
        return 1
    if opp // 3 == _far_row(1 - s) and not _can_capture(board, s, opp):
        return 2                                   # the opponent's "try" succeeded
    return 0


@lru_cache(maxsize=500_000)
def children(key):
    """Tuple of (action, child_key) for a non-terminal position, else ()."""
    if terminal_code(key):
        return ()
    board, hands, s = decode_key(key)
    out = []
    for sq, code in enumerate(board):
        if not code or (code >> 3) != s:
            continue
        t = code & 7
        r, c = divmod(sq, 3)
        for dr, dc in STEPS[t]:
            ar, ac = (-dr, dc) if s else (dr, dc)  # absolute direction
            nr, nc = r + ar, c + ac
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            dst = nr * 3 + nc
            target = board[dst]
            if target and (target >> 3) == s:
                continue
            nb, nh = board[:], [h[:] for h in hands]
            nb[sq] = 0
            if target:
                ct = target & 7
                if ct != LION:
                    nh[s][(CHICK if ct == HEN else ct) - 1] += 1
            nt = HEN if (t == CHICK and nr == _far_row(s)) else t   # mandatory promotion
            nb[dst] = nt | (s << 3)
            rel_dir = (dr, dc) if not s else (dr, -dc)  # the board is rotated by 180 degrees for the second player
            action = _rel_sq(sq, s) * 8 + DIRS.index(rel_dir)
            out.append((action, encode_key(nb, nh, 1 - s)))
    for t in range(3):
        if hands[s][t]:
            for sq in range(12):
                if not board[sq]:
                    nb, nh = board[:], [h[:] for h in hands]
                    nh[s][t] -= 1
                    nb[sq] = (t + 1) | (s << 3)
                    out.append((96 + t * 12 + _rel_sq(sq, s), encode_key(nb, nh, 1 - s)))
    return tuple(out)


def _build_mirror_perm():
    perm = np.zeros(NUM_ACTIONS, np.int64)
    for sq in range(12):
        r, c = divmod(sq, 3)
        msq = r * 3 + (2 - c)
        for d, (dr, dc) in enumerate(DIRS):
            perm[sq * 8 + d] = msq * 8 + DIRS.index((dr, -dc))
        for t in range(3):
            perm[96 + t * 12 + sq] = 96 + t * 12 + msq
    return perm


MIRROR_PERM = _build_mirror_perm()


class Dobutsu(Game):
    num_actions = NUM_ACTIONS
    planes = 16

    def initial_state(self):
        k = initial_key()
        return (k, (k,))

    def legal_moves(self, state):
        key, hist = state
        if self.result(state) is not None:
            return []
        return [a for a, _ in children(key)]

    def play(self, state, action):
        key, hist = state
        for a, child in children(key):
            if a == action:
                return (child, hist + (child,))
        raise ValueError(f"illegal action {action}")

    def result(self, state):
        key, hist = state
        tc = terminal_code(key)
        if tc:
            return 1 if tc == 1 else -1
        if hist.count(key) >= 3 or len(hist) > MAX_PLIES:
            return 0  # threefold repetition (or length cap)
        return None

    # ----- network interface -----------------------------------------------------------------
    def encode(self, state):
        """16 planes of 4x3 from the mover's point of view: own pieces (5 types), opponent pieces
        (5 types), own hand (3 types, count/2), opponent hand (3 types)."""
        board, hands, s = decode_key(state[0])
        out = np.zeros((self.planes, ROWS, COLS), np.float32)
        for sq, code in enumerate(board):
            if code:
                rs = _rel_sq(sq, s)
                owner_is_mover = (code >> 3) == s
                out[(0 if owner_is_mover else 5) + (code & 7) - 1, rs // 3, rs % 3] = 1.0
        for t in range(3):
            out[10 + t] = hands[s][t] / 2.0
            out[13 + t] = hands[1 - s][t] / 2.0
        return out

    def symmetries(self, planes, pi):
        mp = np.zeros_like(pi)
        mp[MIRROR_PERM] = pi
        return [(planes, pi), (planes[:, :, ::-1].copy(), mp)]

    def render(self, state):
        board, hands, s = decode_key(state[0])
        lines = []
        for r in range(ROWS - 1, -1, -1):
            row = ""
            for c in range(COLS):
                code = board[r * 3 + c]
                ch = PIECE_CHARS[code & 7] if code else "."
                row += (ch.upper() if (code >> 3) == 0 else ch) + " " if code else ". "
            lines.append(row.rstrip())
        names = "cge"
        h0 = "".join(names[t] * hands[0][t] for t in range(3)) or "-"
        h1 = "".join(names[t] * hands[1][t] for t in range(3)) or "-"
        lines.append(f"hand UPPER(first): {h0} | hand lower(second): {h1} | to move: {'UPPER' if s == 0 else 'lower'}")
        return "\n".join(lines)
