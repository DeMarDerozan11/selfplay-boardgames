import random
import pytest
from src.games.connect4 import Connect4

g = Connect4()


def play_seq(cols):
    s = g.initial_state()
    for c in cols:
        assert g.result(s) is None
        s = g.play(s, c)
    return s


def test_initial():
    s = g.initial_state()
    assert g.legal_moves(s) == list(range(7))
    assert g.result(s) is None


def test_gravity_and_render():
    s = play_seq([3, 3, 3])
    assert g.cell(s, 3, 0) == "X" and g.cell(s, 3, 1) == "O" and g.cell(s, 3, 2) == "X"


def test_does_not_mutate():
    s = g.initial_state()
    g.play(s, 0)
    assert s == (0, 0, 0)


def test_vertical_win():
    s = play_seq([0, 1, 0, 1, 0, 1, 0])  # X plays 0 four times
    assert g.result(s) == -1  # player to move (O) has lost
    assert g.legal_moves(s) == []


def test_horizontal_win_second_player():
    s = play_seq([0, 1, 0, 2, 6, 3, 6, 4])  # O: 1,2,3,4 on bottom row
    assert g.result(s) == -1 and g.cell(s, 1, 0) == "O"


def test_diagonal_up_right():
    # X on (0,0),(1,1),(2,2),(3,3); O fills around
    s = play_seq([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
    assert g.result(s) == -1 and g.cell(s, 3, 3) == "X"


def test_diagonal_up_left():
    # X on (3,0),(2,1),(1,2),(0,3)
    s = play_seq([3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0])
    assert g.result(s) == -1 and g.cell(s, 0, 3) == "X"


def test_no_wrap_between_columns():
    # X owns (0,5) top of col 0 and (1,0),(1,1),(1,2) bottom of col 1: not a line of 4.
    # col 0 pattern XXOOOX is bottom-up: X,O,... filled alternately with col 1 moves
    s = play_seq([0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2])
    assert g.result(s) is None


def test_full_column_illegal():
    s = play_seq([0, 0, 0, 0, 0, 0])
    assert 0 not in g.legal_moves(s)
    with pytest.raises(ValueError):
        g.play(s, 0)


def test_draw():
    s = play_seq([4, 3, 6, 0, 1, 4, 5, 5, 1, 1, 5, 0, 1, 6, 0, 1, 5, 5, 1, 0, 4, 6, 3, 2, 6, 6, 0, 4, 6, 5, 2, 0, 4, 2, 4, 2, 2, 2, 3, 3, 3, 3])
    assert g.result(s) == 0 and s[2] == 42 and g.legal_moves(s) == []


def test_random_games_consistent():
    """Cross-check the bitboard win detection against a naive board scan."""
    rng = random.Random(0)
    for _ in range(300):
        s = g.initial_state()
        while g.result(s) is None:
            s = g.play(s, rng.choice(g.legal_moves(s)))
        grid = [[g.cell(s, c, r) for c in range(7)] for r in range(6)]
        assert (g.result(s) == -1) == naive_win(grid)
        if g.result(s) == 0:
            assert s[2] == 42


def naive_win(grid):
    for r in range(6):
        for c in range(7):
            if grid[r][c] == ".":
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(0 <= r + k*dr < 6 and 0 <= c + k*dc < 7 and grid[r+k*dr][c+k*dc] == grid[r][c]
                       for k in range(4)):
                    return True
    return False
