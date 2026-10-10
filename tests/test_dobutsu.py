import pathlib
import random
import subprocess

import numpy as np
import pytest

from src.games import dobutsu as D
from src.games.dobutsu import Dobutsu, CHICK, GIRAFFE, ELEPHANT, LION, HEN, encode_key, decode_key

g = Dobutsu()
SOLVER = pathlib.Path(__file__).resolve().parents[1] / "external" / "dobutsu_solver"


def make(pieces, hands=((0, 0, 0), (0, 0, 0)), side=0):
    """pieces: {(row, col): (owner, type)} -> state (history = just this position)."""
    board = [0] * 12
    for (r, c), (o, t) in pieces.items():
        board[r * 3 + c] = t | (o << 3)
    k = encode_key(board, [list(h) for h in hands], side)
    return (k, (k,))


def action_squares(side, a):
    """Absolute (from, to) squares of a move action, derived from geometry only."""
    rel_from, (dr, dc) = a // 8, D.DIRS[a % 8]
    rel_to = rel_from + dr * 3 + dc
    assert 0 <= rel_from // 3 + dr < 4 and 0 <= rel_from % 3 + dc < 3
    return (rel_from, rel_to) if side == 0 else (11 - rel_from, 11 - rel_to)


def act(state, frm, to):
    """Action of the legal move from (row, col) to (row, col), absolute coordinates."""
    side = decode_key(state[0])[2]
    for a in g.legal_moves(state):
        if a < 96 and action_squares(side, a) == (frm[0] * 3 + frm[1], to[0] * 3 + to[1]):
            return a
    raise AssertionError("no such move")


def test_start_position():
    s = g.initial_state()
    assert sorted(g.legal_moves(s)) and len(g.legal_moves(s)) == 4  # lion x2, giraffe x1, chick takes chick
    assert g.result(s) is None
    print(g.render(s))


def test_hen_moves_like_gold_and_chick_promotes():
    # first player's chick one step from the far row, far lion pieces out of the way
    s = make({(2, 0): (0, CHICK), (0, 1): (0, LION), (3, 1): (1, LION)}, side=0)
    s2 = g.play(s, act(s, (2, 0), (3, 0)))
    board, _, _ = decode_key(s2[0])
    assert board[9] == (HEN | 0)  # promoted (mandatory)
    # a hen in the middle has exactly 6 moves (gold general): all except the two backward diagonals
    s = make({(1, 1): (0, HEN), (0, 0): (0, LION), (3, 2): (1, LION)}, side=0)
    hen_moves = [a for a in g.legal_moves(s) if a < 96 and (a // 8) == 4]
    assert len(hen_moves) == 6


def test_capture_goes_to_hand_hen_comes_back_as_chick():
    s = make({(1, 1): (0, GIRAFFE), (2, 1): (1, HEN), (0, 0): (0, LION), (3, 2): (1, LION)}, side=0)
    s2 = g.play(s, act(s, (1, 1), (2, 1)))
    _, hands, _ = decode_key(s2[0])
    assert hands[0] == [1, 0, 0]  # captured hen is a chick in hand


def test_drop_chick_on_far_row_stays_chick_and_has_no_moves():
    s = make({(0, 0): (0, LION), (2, 2): (1, LION)}, hands=((1, 0, 0), (0, 0, 0)), side=0)
    drops = [a for a in g.legal_moves(s) if a >= 96]
    assert len(drops) == 10  # 12 squares - 2 occupied, no restriction on drops
    far = 96 + 0 * 12 + 10  # chick dropped on (3,1)
    s2 = g.play(s, far)
    board, _, _ = decode_key(s2[0])
    assert board[10] == CHICK
    s3 = make({(3, 1): (0, CHICK), (0, 0): (0, LION), (2, 2): (1, LION)}, side=0)
    assert not any(a < 96 and a // 8 == 10 for a in g.legal_moves(s3))  # stuck chick cannot move


def test_capturing_lion_wins():
    s = make({(1, 1): (0, GIRAFFE), (2, 1): (1, LION), (0, 0): (0, LION)}, side=0)
    s2 = g.play(s, act(s, (1, 1), (2, 1)))
    assert g.result(s2) == -1  # the player to move (second) has lost
    assert g.legal_moves(s2) == []


def test_try_wins_if_lion_cannot_be_captured():
    s = make({(2, 1): (0, LION), (1, 2): (1, LION)}, side=0)  # first player's lion steps to row 3, out of reach
    s2 = g.play(s, act(s, (2, 1), (3, 1)))
    assert g.result(s2) == -1


def test_try_fails_if_lion_can_be_captured_immediately():
    s = make({(2, 1): (0, LION), (3, 0): (1, LION), (3, 2): (1, GIRAFFE)}, side=0)
    # lion to (3,1) is adjacent to the opponent's lion and giraffe -> capturable, game goes on
    s2 = g.play(s, act(s, (2, 1), (3, 1)))
    assert g.result(s2) is None
    s3 = g.play(s2, act(s2, (3, 0), (3, 1)))  # opponent captures the lion
    assert g.result(s3) == -1


def test_lion_on_far_row_that_survives_wins_for_its_owner():
    # try that could be captured, but the opponent does not capture -> owner wins at its turn
    s = make({(2, 1): (0, LION), (3, 0): (1, LION), (3, 2): (1, GIRAFFE)}, side=0)
    s2 = g.play(s, act(s, (2, 1), (3, 1)))
    s3 = g.play(s2, act(s2, (3, 2), (2, 2)))  # giraffe steps away instead of capturing
    assert g.result(s3) == 1  # first player to move, own lion survived on the far row: wins


def test_threefold_repetition_is_a_draw():
    s = make({(0, 0): (0, LION), (3, 2): (1, LION)}, side=0)
    for _ in range(2):
        s = g.play(s, act(s, (0, 0), (0, 1)))
        s = g.play(s, act(s, (3, 2), (3, 1)))
        s = g.play(s, act(s, (0, 1), (0, 0)))
        s = g.play(s, act(s, (3, 1), (3, 2)))
    assert g.result(s) == 0


def random_states(n, seed=0):
    rng, out = random.Random(seed), []
    while len(out) < n:
        s = g.initial_state()
        for _ in range(rng.randint(0, 60)):
            if g.result(s) is not None:
                break
            s = g.play(s, rng.choice(g.legal_moves(s)))
            out.append(s)
    return out[:n]


def test_non_terminal_positions_always_have_a_move():
    for s in random_states(3000):
        if g.result(s) is None:
            assert g.legal_moves(s)


def mirror_key(k):
    board, hands, side = decode_key(k)
    nb = [board[r * 3 + (2 - c)] for r in range(4) for c in range(3)]
    return encode_key(nb, hands, side)


def test_mirror_symmetry_of_encoding_and_actions():
    for s in random_states(300, seed=1):
        if g.result(s) is not None:
            continue
        m = (mirror_key(s[0]), (mirror_key(s[0]),))
        planes, mplanes = g.encode(s), g.encode(m)
        assert np.array_equal(planes[:, :, ::-1], mplanes)
        assert sorted(D.MIRROR_PERM[a] for a in g.legal_moves(s)) == sorted(g.legal_moves(m))
    assert np.array_equal(D.MIRROR_PERM[D.MIRROR_PERM], np.arange(D.NUM_ACTIONS))


def test_actions_are_unique_per_position():
    for s in random_states(500, seed=2):
        acts = [a for a, _ in D.children(s[0])]
        assert len(acts) == len(set(acts)) and all(0 <= a < D.NUM_ACTIONS for a in acts)


@pytest.mark.skipif(not SOLVER.exists(), reason="run scripts/build_dobutsu_solver.sh first")
def test_rules_agree_with_c_solver():
    """The independent C++ move generator and the Python one must produce identical children."""
    states = random_states(1500, seed=3)
    keys = sorted({s[0] for s in states})
    out = subprocess.run([str(SOLVER), "children"], input="\n".join(map(str, keys)) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(keys)
    for k, line in zip(keys, out):
        parts = [int(x) for x in line.split()]
        assert parts[0] == D.terminal_code(k), k
        assert parts[1:] == sorted({c for _, c in D.children(k)}), k


def test_action_geometry_matches_resulting_board():
    for s in random_states(400, seed=4):
        if g.result(s) is not None:
            continue
        board, _, side = decode_key(s[0])
        for a, child in D.children(s[0]):
            if a >= 96:
                continue
            frm, to = action_squares(side, a)
            cb, _, _ = decode_key(child)
            assert board[frm] and (board[frm] >> 3) == side and cb[frm] == 0 and cb[to] != 0
            assert all(cb[i] == board[i] for i in range(12) if i not in (frm, to))
