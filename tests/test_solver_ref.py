import pytest
from src.solver_ref import BINARY, solve_many, best_value

pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="run scripts/build_solver.sh first")


def test_immediate_win_scores():
    # X has 3 in a row on the bottom (cols 2,3,4 -> 0-indexed), O above; X to move wins at col 1 or 5
    sc = solve_many([(2, 2, 3, 3, 4, 4)])[(2, 2, 3, 3, 4, 4)]
    assert sc[1] == sc[5] == best_value(sc) == 18  # (42 + 1 - 6) // 2: win with the 7th stone
    assert sc[1] > 0
