# cfr_opponent.py
# CFR utilities: build fixed opponent + sample CFR actions.

import numpy as np
from open_spiel.python.algorithms import cfr as cfr_lib


def build_cfr_average_policy(game, iters: int):
    """
    Train a CFR solver for `iters` iterations and return its average policy.
    Used as a strong fixed opponent.
    """
    solver = cfr_lib.CFRSolver(game)
    for _ in range(iters):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def cfr_action_from_state(state, player_id: int, cfr_policy) -> int:
    """
    Sample an action from CFR policy given a pyspiel.State.
    """
    ap = cfr_policy.action_probabilities(state, player_id)
    if not ap:
        return 0

    actions, probs = zip(*ap.items())
    probs = np.asarray(probs, dtype=np.float64)

    s = probs.sum()
    if s <= 0:
        return int(np.random.choice(actions))

    probs = probs / s
    return int(np.random.choice(actions, p=probs))