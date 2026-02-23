# eval.py
from __future__ import annotations

from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch

from agents import masked_softmax
from config import ACT_DIM, EVAL_EPISODES, REGRET_ROOT_SIMS, REGRET_ROLLOUT_HORIZON


def _policy_action(model, obs: np.ndarray, legal: List[int], device: str, sample: bool = True) -> int:
    obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    mask = torch.zeros((1, ACT_DIM), dtype=torch.float32, device=device)
    if len(legal) > 0:
        mask[0, torch.tensor(legal, dtype=torch.long, device=device)] = 1.0

    with torch.no_grad():
        logits, _ = model(obs_t)
        probs = masked_softmax(logits, mask)  # [1, A]
        if sample:
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())
        else:
            return int(torch.argmax(probs, dim=-1).item())


def _rollout_episode(env, model, device: str, starting_player: int) -> int:
    """
    Returns outcome: +1 if X wins, -1 if O wins, 0 draw.
    """
    env.reset(starting_player=starting_player)
    while True:
        legal = env.legal_actions()
        a = _policy_action(model, env._get_obs(), legal, device, sample=True)
        step = env.step(a)
        if step.done:
            return int(step.info.get("winner", 0))


def eval_reward_and_regret(env_cls, model, device: str) -> Tuple[float, float, float, float]:
    """
    reward_p0: average outcome when policy plays as X (starting_player=+1)
    reward_p1: average outcome when policy plays as O (starting_player=-1)
    regret_p0/regret_p1: root-only best-action improvement (approx regret)
      - For seat X: choose best first move by MC, then follow policy
      - For seat O: same, but starting_player=-1
    """
    # -------- reward (policy value) --------
    outcomes_x = []
    outcomes_o = []
    for _ in range(EVAL_EPISODES):
        env = env_cls()
        outcomes_x.append(_rollout_episode(env, model, device, starting_player=+1))
        env = env_cls()
        outcomes_o.append(_rollout_episode(env, model, device, starting_player=-1))

    # outcome is winner (+1/-1/0), reward for X seat is that outcome
    reward_p0 = float(np.mean(outcomes_x))
    reward_p1 = float(np.mean(outcomes_o))  # this is still winner sign; interpret separately if you want

    # -------- regret (root-only best action improvement) --------
    regret_p0 = _root_only_regret(env_cls, model, device, starting_player=+1)
    regret_p1 = _root_only_regret(env_cls, model, device, starting_player=-1)

    return reward_p0, reward_p1, float(regret_p0), float(regret_p1)


def _simulate_from_state(env, model, device: str, horizon: int) -> int:
    """
    Simulate (policy vs itself) until done or horizon.
    Returns winner sign (+1/-1/0).
    """
    for _ in range(horizon):
        legal = env.legal_actions()
        if len(legal) == 0:
            break
        a = _policy_action(model, env._get_obs(), legal, device, sample=True)
        step = env.step(a)
        if step.done:
            return int(step.info.get("winner", 0))
    return int(getattr(env, "winner", 0))


def _root_only_regret(env_cls, model, device: str, starting_player: int) -> float:
    """
    Approx regret = V(best first move) - V(policy first move)
    where value is expected winner sign (+1/-1/0) under policy afterwards.
    """
    env0 = env_cls()
    obs0 = env0.reset(starting_player=starting_player)
    legal0 = env0.legal_actions()

    if len(legal0) == 0:
        return 0.0

    # Baseline: policy chooses first move (sample) then follow policy
    base_vals = []
    for _ in range(REGRET_ROOT_SIMS):
        env = env_cls()
        env.reset(starting_player=starting_player)
        a0 = _policy_action(model, env._get_obs(), env.legal_actions(), device, sample=True)
        env.step(a0)
        w = _simulate_from_state(env, model, device, REGRET_ROLLOUT_HORIZON)
        base_vals.append(w)
    v_policy = float(np.mean(base_vals))

    # Best first move by MC: try every legal action, estimate continuation value
    best_v = -1e9
    for a in legal0:
        vals = []
        for _ in range(REGRET_ROOT_SIMS):
            env = env_cls()
            env.reset(starting_player=starting_player)
            env.step(a)
            w = _simulate_from_state(env, model, device, REGRET_ROLLOUT_HORIZON)
            vals.append(w)
        v = float(np.mean(vals))
        if v > best_v:
            best_v = v

    return max(0.0, best_v - v_policy)