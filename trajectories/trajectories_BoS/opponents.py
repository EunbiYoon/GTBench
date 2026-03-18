"""
opponents.py — Typed opponent pool for Battle of the Sexes.

Each opponent has a deterministic base policy + epsilon-noise for diversity.
This implements Strategy D from §6.5.4 of the training doc.
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Sequence
from game import Action, ACTIONS


class Opponent(ABC):
    name: str
    description: str

    def __init__(self, epsilon: float = 0.15, seed: int | None = None):
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def act(self, history: list[Action]) -> Action:
        """Return an action, with epsilon chance of random deviation."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        return self._base_act(history)

    @abstractmethod
    def _base_act(self, history: list[Action]) -> Action:
        """Deterministic base policy."""
        ...

    def reset(self):
        """Reset any internal state between episodes."""
        pass


# ---------------------------------------------------------------------------
# Concrete opponent types
# ---------------------------------------------------------------------------

class AlwaysOpera(Opponent):
    name = "always_opera"
    description = "Always chooses Opera regardless of history"

    def _base_act(self, history: list[Action]) -> Action:
        return "Opera"


class AlwaysFootball(Opponent):
    name = "always_football"
    description = "Always chooses Football regardless of history"

    def _base_act(self, history: list[Action]) -> Action:
        return "Football"


class TitForTat(Opponent):
    """Starts with Opera, then copies P1's last move."""
    name = "tit_for_tat"
    description = "Starts Opera, then mirrors P1's last action"

    def _base_act(self, history: list[Action]) -> Action:
        if not history:
            return "Opera"
        return history[-1]


class Grudger(Opponent):
    """Plays Opera until P1 ever plays Football, then always Football."""
    name = "grudger"
    description = "Cooperates (Opera) until P1 plays Football, then always Football"

    def __init__(self, epsilon: float = 0.15, seed: int | None = None):
        super().__init__(epsilon, seed)
        self._triggered = False

    def reset(self):
        self._triggered = False

    def _base_act(self, history: list[Action]) -> Action:
        if not self._triggered and "Football" in history:
            self._triggered = True
        return "Football" if self._triggered else "Opera"


class InitialYielder(Opponent):
    """Yields to P1's preferred (Opera) for first N rounds, then switches to Football."""
    name = "initial_yielder"
    description = "Plays Opera for the first yield_rounds, then always Football"

    def __init__(self, yield_rounds: int = 2, epsilon: float = 0.15, seed: int | None = None):
        super().__init__(epsilon, seed)
        self.yield_rounds = yield_rounds

    def _base_act(self, history: list[Action]) -> Action:
        return "Opera" if len(history) < self.yield_rounds else "Football"


class MajorityRule(Opponent):
    """Plays whichever action P1 used most often in history. Ties -> Opera."""
    name = "majority_rule"
    description = "Plays whichever action P1 played most often in history"

    def _base_act(self, history: list[Action]) -> Action:
        if not history:
            return "Opera"
        opera_count = history.count("Opera")
        football_count = history.count("Football")
        return "Opera" if opera_count >= football_count else "Football"


class RandomOpponent(Opponent):
    """Fully random — useful as a baseline / sanity check."""
    name = "random"
    description = "Chooses uniformly at random each round"

    def _base_act(self, history: list[Action]) -> Action:
        return self._rng.choice(ACTIONS)


# ---------------------------------------------------------------------------
# Pool factory
# ---------------------------------------------------------------------------

OPPONENT_REGISTRY: dict[str, type[Opponent]] = {
    "always_opera": AlwaysOpera,
    "always_football": AlwaysFootball,
    "tit_for_tat": TitForTat,
    "grudger": Grudger,
    "initial_yielder": InitialYielder,
    "majority_rule": MajorityRule,
    "random": RandomOpponent,
}


def build_opponent_pool(
    opponent_cfgs: list[dict],
    epsilon: float = 0.15,
    seed: int | None = None,
) -> list[Opponent]:
    """
    Build a list of Opponent instances from the config.

    Args:
        opponent_cfgs: list of dicts from config.yaml (opponents.types)
        epsilon: noise level applied to all opponents
        seed: base random seed (each opponent gets seed+i for reproducibility)
    """
    pool = []
    for i, cfg in enumerate(opponent_cfgs):
        opp_name = cfg["name"]
        cls = OPPONENT_REGISTRY.get(opp_name)
        if cls is None:
            raise ValueError(f"Unknown opponent type: {opp_name!r}. "
                             f"Available: {list(OPPONENT_REGISTRY)}")

        opp_seed = None if seed is None else seed + i

        # Some opponents take extra kwargs
        extra = {}
        if opp_name == "initial_yielder" and "yield_rounds" in cfg:
            extra["yield_rounds"] = cfg["yield_rounds"]

        pool.append(cls(epsilon=epsilon, seed=opp_seed, **extra))

    return pool
