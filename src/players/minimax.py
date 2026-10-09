"""Depth-limited negamax with alpha-beta pruning, move ordering and a transposition table.

Works on any `Game`. Scores are integers from the point of view of the player to move.
A win found with `d` plies of remaining depth scores WIN + d (so faster wins are preferred
and the score does not depend on the path, which keeps the transposition table sound).
"""
import random

WIN = 10_000
EXACT, LOWER, UPPER = 0, 1, 2


class MinimaxPlayer:
    def __init__(self, depth=4, seed=None, use_tt=True, use_ordering=True):
        self.depth = depth
        self.name = f"minimax-d{depth}"
        self.rng = random.Random(seed)
        self.use_tt, self.use_ordering = use_tt, use_ordering
        self.nodes = 0

    def choose(self, game, state):
        """Pick the best move; ties are broken randomly (so that matches are not deterministic)."""
        scores = self.move_scores(game, state)
        best = max(scores.values())
        return self.rng.choice([m for m, v in scores.items() if v == best])

    def move_scores(self, game, state):
        """Exact (full-window) score of every legal move; slower than needed but unbiased."""
        self.tt = {}
        self.game = game
        self.nodes = 0
        return {m: -self._search(game.play(state, m), self.depth - 1, -WIN * 2, WIN * 2)
                for m in game.legal_moves(state)}

    def _search(self, state, depth, alpha, beta):
        self.nodes += 1
        game = self.game
        r = game.result(state)
        if r is not None:
            return r * (WIN + depth)
        if depth <= 0:
            return game.heuristic(state)
        alpha0 = alpha
        if self.use_tt:
            e = self.tt.get(state)
            if e is not None and e[0] >= depth:
                _, flag, val = e
                if flag == EXACT:
                    return val
                if flag == LOWER:
                    alpha = max(alpha, val)
                else:
                    beta = min(beta, val)
                if alpha >= beta:
                    return val
        moves = game.legal_moves(state)
        if self.use_ordering:
            moves = game.order_moves(state, moves)
        best = -WIN * 3
        for m in moves:
            best = max(best, -self._search(game.play(state, m), depth - 1, -beta, -alpha))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        if self.use_tt:
            flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
            self.tt[state] = (depth, flag, best)
        return best
