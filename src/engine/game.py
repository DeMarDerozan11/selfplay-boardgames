"""Generic game interface. Algorithms (minimax, MCTS, AlphaZero) only use this API."""
from abc import ABC, abstractmethod
from typing import Hashable, Optional


class Game(ABC):
    """A two-player, zero-sum, perfect-information game with immutable states.

    Values are always from the point of view of the player TO MOVE in `state`:
    +1 win, 0 draw, -1 loss.
    """

    num_actions: int  # size of the (fixed) action space

    @abstractmethod
    def initial_state(self) -> Hashable: ...

    @abstractmethod
    def legal_moves(self, state) -> list[int]: ...

    @abstractmethod
    def play(self, state, move: int):
        """Return the new state (the input is never mutated)."""

    @abstractmethod
    def result(self, state) -> Optional[int]:
        """None if the game goes on, else the value for the player to move."""

    @abstractmethod
    def render(self, state) -> str: ...
