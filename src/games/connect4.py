"""Connect 4 (7 columns x 6 rows) on bitboards.

State = (position, mask, moves):
  position: stones of the player TO MOVE
  mask:     stones of both players
  moves:    number of moves played
Bit layout: column c, row r (0 = bottom) -> bit c*7 + r. Row 6 of each column is an
always-empty sentinel so that shifted alignments never wrap into the next column.
Same encoding as Pascal Pons' solver, which makes cross-checking easy.
"""
import numpy as np

from src.engine.game import Game

W, H = 7, 6
H1 = H + 1
BOTTOM = sum(1 << (c * H1) for c in range(W))
BOARD = BOTTOM * ((1 << H) - 1)


def _windows():
    out = []
    for c in range(W):
        for r in range(H):
            for dc, dr in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if all(0 <= x < W and 0 <= y < H for x, y in cells):
                    out.append(sum(1 << (x * H1 + y) for x, y in cells))
    return out


WINDOWS = _windows()  # the 69 possible lines of four
WINDOW_SCORE = (0, 1, 4, 16)  # value of a window holding 0..3 stones of one side only
_CENTER_FIRST = sorted(range(W), key=lambda c: abs(c - W // 2))


def _has_four(p: int) -> bool:
    for shift in (1, H1, H1 - 1, H1 + 1):  # vertical, horizontal, two diagonals
        m = p & (p >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


class Connect4(Game):
    num_actions = W

    def initial_state(self):
        return (0, 0, 0)

    def legal_moves(self, state):
        _, mask, _ = state
        if self.result(state) is not None:
            return []
        top = 1 << (H - 1)
        return [c for c in range(W) if not mask & (top << (c * H1))]

    def play(self, state, move):
        pos, mask, n = state
        if mask & (1 << (H - 1) << (move * H1)):
            raise ValueError(f"column {move} is full")
        new_mask = mask | (mask + (1 << (move * H1)))
        return (pos ^ mask, new_mask, n + 1)  # swap sides: new "position" = opponent's stones

    def result(self, state):
        pos, mask, n = state
        if _has_four(pos ^ mask):  # stones of the player who just moved
            return -1
        return 0 if n == W * H else None

    def order_moves(self, state, moves):
        return sorted(moves, key=_CENTER_FIRST.index)

    def heuristic(self, state):
        """Sum over open windows (containing stones of one side only): mine minus opponent's."""
        pos, mask, _ = state
        opp = pos ^ mask
        score = 0
        for w in WINDOWS:
            a, b = (pos & w).bit_count(), (opp & w).bit_count()
            if b == 0:
                score += WINDOW_SCORE[a]
            elif a == 0:
                score -= WINDOW_SCORE[b]
        return score

    def encode(self, state):
        """Planes: [stones of player to move, stones of opponent], each 6 x 7 (row 0 = bottom)."""
        pos, mask, _ = state
        out = np.zeros((2, H, W), np.float32)
        for plane, bits in ((0, pos), (1, pos ^ mask)):
            for c in range(W):
                col = (bits >> (c * H1)) & ((1 << H) - 1)
                for r in range(H):
                    out[plane, r, c] = (col >> r) & 1
        return out

    def symmetries(self, planes, pi):
        return [(planes, pi), (planes[:, :, ::-1].copy(), pi[::-1].copy())]  # mirror left-right

    def cell(self, state, c, r):
        """'X' = first player, 'O' = second player, '.' empty."""
        pos, mask, n = state
        bit = 1 << (c * H1 + r)
        if not mask & bit:
            return "."
        to_move_is_first = n % 2 == 0
        mine = bool(pos & bit)
        return "X" if mine == to_move_is_first else "O"

    def render(self, state):
        rows = ["".join(self.cell(state, c, r) + " " for c in range(W)).rstrip()
                for r in range(H - 1, -1, -1)]
        return "\n".join(rows + ["-" * (2 * W - 1), " ".join(map(str, range(W)))])
