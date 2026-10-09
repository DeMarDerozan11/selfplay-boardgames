"""Plain MCTS (UCT) with random rollouts, no neural network. Works on any `Game`.

Each node stores N (visits) and W (sum of values) from the point of view of the player
TO MOVE at that node. A parent therefore scores a child with -W/N, and values flip sign
at every level during backpropagation.
"""
import math
import random


class Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "N", "W", "terminal")

    def __init__(self, game, state, parent=None, move=None):
        self.state, self.parent, self.move = state, parent, move
        self.terminal = game.result(state)  # None while the game goes on
        self.untried = [] if self.terminal is not None else game.legal_moves(state)
        self.children = []
        self.N, self.W = 0, 0.0


class MCTSPlayer:
    def __init__(self, simulations=500, c=1.4, seed=None):
        self.simulations, self.c = simulations, c
        self.name = f"mcts-{simulations}"
        self.rng = random.Random(seed)

    def choose(self, game, state):
        visits = self.search(game, state)
        best = max(visits.values())
        return self.rng.choice([m for m, n in visits.items() if n == best])

    def search(self, game, state):
        """Run the simulations and return {move: visit count} at the root."""
        root = Node(game, state)
        for _ in range(self.simulations):
            node = root
            # 1. selection: descend while fully expanded
            while not node.untried and node.children:
                node = self._best_child(node)
            # 2. expansion: add one new child
            if node.untried:
                move = node.untried.pop(self.rng.randrange(len(node.untried)))
                child = Node(game, game.play(node.state, move), node, move)
                node.children.append(child)
                node = child
            # 3. simulation: random playout, value for the player to move at `node`
            value = node.terminal if node.terminal is not None else self._rollout(game, node.state)
            # 4. backpropagation: flip the sign at each level
            while node is not None:
                node.N += 1
                node.W += value
                value = -value
                node = node.parent
        return {ch.move: ch.N for ch in root.children}

    def _best_child(self, node):
        log_n = math.log(node.N)
        return max(node.children,
                   key=lambda ch: -ch.W / ch.N + self.c * math.sqrt(log_n / ch.N))

    def _rollout(self, game, state):
        sign = 1
        while True:
            r = game.result(state)
            if r is not None:
                return r * sign
            state = game.play(state, self.rng.choice(game.legal_moves(state)))
            sign = -sign
