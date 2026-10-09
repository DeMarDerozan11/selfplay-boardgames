"""AlphaZero-style MCTS (PUCT) guided by a policy/value network, with batched leaf evaluation.

Per-edge statistics are stored in numpy arrays of size `num_actions`:
  P: prior from the network, N: visit count, W: total value, all W from the point of view of
  the player to move AT THAT NODE (the player choosing the edge). Values flip sign going up.
Many trees (one per concurrent game) are searched together so the network sees big batches.
"""
import math

import numpy as np
import torch


class Node:
    __slots__ = ("state", "legal", "P", "N", "W", "children", "expanded", "terminal")

    def __init__(self, game, state):
        self.state = state
        self.terminal = game.result(state)
        self.expanded = False
        self.children = [None] * game.num_actions
        a = game.num_actions
        self.P, self.N, self.W = np.zeros(a, np.float32), np.zeros(a, np.float32), np.zeros(a, np.float32)
        self.legal = np.zeros(a, bool)
        if self.terminal is None:
            self.legal[game.legal_moves(state)] = True


class Tree:
    def __init__(self, game, state, c_puct=1.5, noise=None, rng=None):
        self.game, self.c_puct, self.noise, self.rng = game, c_puct, noise, rng
        self.root = Node(game, state)
        self.sims = 0

    # --- selection -------------------------------------------------------------------------
    def _select(self, node):
        q = np.divide(node.W, node.N, out=np.zeros_like(node.W), where=node.N > 0)
        u = q + self.c_puct * node.P * math.sqrt(node.N.sum() + 1.0) / (1.0 + node.N)
        u[~node.legal] = -np.inf
        return int(np.argmax(u))

    def descend(self):
        """Walk down until a leaf needing a network evaluation (returned) or a terminal state
        (backed up immediately, returns None)."""
        if not self.root.expanded:
            return self.root, []
        node, path = self.root, []
        while True:
            a = self._select(node)
            path.append((node, a))
            child = node.children[a]
            if child is None:
                child = node.children[a] = Node(self.game, self.game.play(node.state, a))
            if child.terminal is not None:
                self._backup(path, child.terminal)
                return None
            if not child.expanded:
                return child, path
            node = child

    def finish(self, leaf, path, priors, value):
        leaf.P = priors * leaf.legal
        leaf.expanded = True
        if not path:  # root: optionally mix in exploration noise (self-play only)
            if self.noise is not None:
                eps, alpha = self.noise
                legal = np.flatnonzero(leaf.legal)
                dir_ = np.zeros_like(leaf.P)
                dir_[legal] = self.rng.dirichlet([alpha] * len(legal))
                leaf.P = (1 - eps) * leaf.P + eps * dir_
        else:
            self._backup(path, value)

    def _backup(self, path, v):
        for node, a in reversed(path):
            v = -v  # leaf value -> point of view of the player who chose the edge
            node.N[a] += 1
            node.W[a] += v
        self.sims += 1


@torch.no_grad()
def evaluate(game, net, states):
    """Batch evaluation. Returns masked+renormalised priors (B, A) and values (B,)."""
    x = torch.from_numpy(np.stack([game.encode(s) for s in states]))
    logits, v = net(x)
    legal = torch.zeros(len(states), game.num_actions, dtype=torch.bool)
    for i, s in enumerate(states):
        legal[i, game.legal_moves(s)] = True
    logits = logits.masked_fill(~legal, -1e9)
    return torch.softmax(logits, dim=1).numpy(), v.numpy()


def run_searches(game, net, trees, sims):
    """Advance all trees until each has done `sims` simulations; leaves evaluated in batches."""
    net.eval()
    while True:
        pending = []
        for t in trees:
            while t.sims < sims:
                p = t.descend()
                if p is not None:
                    pending.append((t, p[0], p[1]))
                    break
        if not pending:
            return
        priors, values = evaluate(game, net, [leaf.state for _, leaf, _ in pending])
        for (t, leaf, path), pr, v in zip(pending, priors, values):
            t.finish(leaf, path, pr, float(v))


def visit_counts(tree):
    return tree.root.N.copy()


def best_moves(game, net, states, sims):
    """Greedy (most visited) move for each state, batched. sims=0 -> raw network policy."""
    if sims == 0:
        priors, _ = evaluate(game, net, states)
        return [int(np.argmax(p)) for p in priors]
    trees = [Tree(game, s) for s in states]
    run_searches(game, net, trees, sims)
    return [int(np.argmax(visit_counts(t))) for t in trees]


class AZPlayer:
    """Plays with network + search, no exploration noise, greedy on visit counts."""

    def __init__(self, net, sims=100, name=None):
        self.net, self.sims = net.eval(), sims
        self.name = name or f"az-{sims}"

    def choose(self, game, state):
        return best_moves(game, self.net, [state], self.sims)[0]
