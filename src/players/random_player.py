import random


class RandomPlayer:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, game, state):
        return self.rng.choice(game.legal_moves(state))
