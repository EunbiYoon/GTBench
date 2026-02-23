# train.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

from agents import masked_softmax
from config import (
    ACT_DIM,
    GAMMA,
    GAE_LAMBDA,
    CLIP_EPS,
    ENT_COEF,
    VF_COEF,
    PPO_EPOCHS,
    MINIBATCH_SIZE,
    MAX_GRAD_NORM,
)


@dataclass
class RolloutBatch:
    obs: torch.Tensor        # [T, obs_dim]
    actions: torch.Tensor    # [T]
    logp: torch.Tensor       # [T]
    values: torch.Tensor     # [T]
    rewards: torch.Tensor    # [T]
    dones: torch.Tensor      # [T]
    players: torch.Tensor    # [T] player who acted (+1 X / -1 O)


def _legal_mask(legal_actions: List[int], act_dim: int, device: str) -> torch.Tensor:
    m = torch.zeros((act_dim,), dtype=torch.float32, device=device)
    if len(legal_actions) > 0:
        m[torch.tensor(legal_actions, dtype=torch.long, device=device)] = 1.0
    return m


def collect_rollout(env, model, act_dim: int, device: str, episode_counter_ref: List[int],
                    debug_print_every_episode: int = 0) -> RolloutBatch:
    """
    Collect ROLLOUT_STEPS transitions in self-play.
    Reward returned by env is for the player who JUST moved (before switching turns).
    We store `player` and later convert terminal outcome into per-step rewards for both players.
    """
    from config import ROLLOUT_STEPS  # avoid circular import

    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, ply_buf = [], [], [], [], [], [], []

    obs = env.reset(starting_player=1)
    ep_index = 0

    for t in range(ROLLOUT_STEPS):
        player = env.current_player
        legal = env.legal_actions()
        if len(legal) == 0:
            # shouldn't happen unless done; reset
            obs = env.reset(starting_player=1)
            player = env.current_player
            legal = env.legal_actions()

        obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # [1, obs_dim]
        mask_t = _legal_mask(legal, act_dim, device).unsqueeze(0)      # [1, act_dim]

        with torch.no_grad():
            logits, value = model(obs_t)
            probs = masked_softmax(logits, mask_t)  # [1, A]
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()                 # [1]
            logp = dist.log_prob(action)           # [1]

        a = int(action.item())
        step = env.step(a)

        # store transition
        obs_buf.append(obs.copy())
        act_buf.append(a)
        logp_buf.append(float(logp.item()))
        val_buf.append(float(value.item()))
        rew_buf.append(float(step.reward))         # reward for the acting player
        done_buf.append(1.0 if step.done else 0.0)
        ply_buf.append(int(player))

        obs = step.observation

        if step.done:
            episode_counter_ref[0] += 1
            ep_index += 1

            if debug_print_every_episode > 0 and (episode_counter_ref[0] % debug_print_every_episode == 0):
                print("\n=== Episode end ===")
                print(f"winner={step.info.get('winner')}, reason={step.info.get('reason')}")
                print(env.render(as_matrix=True))
                print("===================\n")

            obs = env.reset(starting_player=1)

    # convert to tensors
    obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
    actions_t = torch.tensor(act_buf, dtype=torch.long, device=device)
    logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
    values_t = torch.tensor(val_buf, dtype=torch.float32, device=device)
    rewards_t = torch.tensor(rew_buf, dtype=torch.float32, device=device)
    dones_t = torch.tensor(done_buf, dtype=torch.float32, device=device)
    players_t = torch.tensor(ply_buf, dtype=torch.int64, device=device)

    return RolloutBatch(obs_t, actions_t, logp_t, values_t, rewards_t, dones_t, players_t)


def _compute_gae(rewards, values, dones, gamma: float, lam: float):
    """
    GAE-Lambda.
    Here rewards are sparse (0 until terminal, then +1 for winner move).
    We'll compute advantages per timestep.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


def ppo_update(model, optimizer, batch: RolloutBatch):
    """
    PPO update on collected rollout.
    NOTE: reward shaping:
      env reward is for acting player only on terminal winning move (+1).
      That is fine; policy learns to create terminal winning moves.
      (If you want symmetric terminal reward for all prior moves, we'd add outcome backfilling.)
    """
    obs = batch.obs
    actions = batch.actions
    old_logp = batch.logp
    old_values = batch.values
    rewards = batch.rewards
    dones = batch.dones

    # Normalize advantages
    adv, returns = _compute_gae(rewards, old_values, dones, GAMMA, GAE_LAMBDA)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.shape[0]
    idx = torch.arange(N, device=obs.device)

    for _ in range(PPO_EPOCHS):
        perm = idx[torch.randperm(N)]
        for start in range(0, N, MINIBATCH_SIZE):
            mb = perm[start:start + MINIBATCH_SIZE]

            logits, values = model(obs[mb])
            # build legal mask from obs by reconstructing empties:
            # obs format: 16 cells + current_player; empty cell => 0
            cells = obs[mb, :ACT_DIM]
            mask = (cells == 0.0).float()  # empty => legal
            probs = masked_softmax(logits, mask)
            dist = torch.distributions.Categorical(probs=probs)

            logp = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns[mb])

            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()