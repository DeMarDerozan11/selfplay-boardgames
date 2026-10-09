"""Console demo: random vs random, or you (X, first) vs random."""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.games.connect4 import Connect4
from src.players.random_player import RandomPlayer
from src.engine.match import play_game, evaluate


class Human:
    name = "human"
    def choose(self, game, state):
        print(game.render(state))
        legal = game.legal_moves(state)
        while True:
            s = input(f"column {legal}: ")
            if s.isdigit() and int(s) in legal:
                return int(s)

ap = argparse.ArgumentParser()
ap.add_argument("--human", action="store_true")
ap.add_argument("--bench", type=int, default=0, help="random vs random, N games")
a = ap.parse_args()
g = Connect4()
if a.bench:
    print(evaluate(g, RandomPlayer(0), RandomPlayer(1), a.bench))
else:
    p1 = Human() if a.human else RandomPlayer(0)
    print("result for first player:", play_game(g, p1, RandomPlayer(1), verbose=not a.human))
