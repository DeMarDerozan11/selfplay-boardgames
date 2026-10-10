"""Game registry: everything the generic training / evaluation scripts need to know about a game."""
from src.games.connect4 import Connect4
from src.games.dobutsu import Dobutsu


def make_game(name):
    """Return (game, net_kwargs) where net_kwargs are the input/output shapes of the network."""
    if name == "connect4":
        return Connect4(), dict(in_ch=2, height=6, width=7, num_actions=7)
    if name == "dobutsu":
        return Dobutsu(), dict(in_ch=Dobutsu.planes, height=4, width=3, num_actions=Dobutsu.num_actions)
    raise ValueError(f"unknown game {name!r}")
