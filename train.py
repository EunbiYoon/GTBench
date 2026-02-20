# train.py
# PPO training utilities: rollout collection + PPO update.

import numpy as np
import torch
import torch.nn as nn

from config import (
    ROLLOUT_STEPS, PRINT_EVERY_EPISODES,
    GAMMA, LAMBDA, CLIP_EPS, VF_COEF, ENT_COEF, MAX_GRAD_NORM,
    EPOCHS, MINIBATCH
)
from agents import ActorCritic, masked_softmax


def collect_rollout(env, model: ActorCritic, act_dim: int, device: str, episode_counter_ref):
    """
    Collect on-policy transitions using self-play in rl_environment.

    Returns tensors of length ROLLOUT_STEPS:
      obs      : [T, obs_dim]
      act      : [T]
      mask     : [T, act_dim]
      logp_old : [T]
      ret      : [T]  (GAE returns)
      adv      : [T]  (normalized advantages)
    """
    obs_list, act_list, mask_list = [], [], []
    logp_list, val_list = [], []
    rew_list, done_list = [], []

    ts = env.reset()
    steps = 0

    while steps < ROLLOUT_STEPS:
        # If an episode ended, count it and start a new one.
        if ts.last():
            episode_counter_ref[0] += 1
            if episode_counter_ref[0] % PRINT_EVERY_EPISODES == 0:
                print(f"[progress] episodes={episode_counter_ref[0]}")
            ts = env.reset()
            continue

        # Player to act (0 or 1). For tic_tac_toe, no chance nodes expected.
        p = int(ts.observations["current_player"])
        if p < 0:
            ts = env.reset()
            continue

        # Observation for current player + legal action mask
        obs = np.asarray(ts.observations["info_state"][p], dtype=np.float32).reshape(-1)
        legal = ts.observations["legal_actions"][p]

        mask = np.zeros((act_dim,), dtype=np.float32)
        for a in legal:
            if 0 <= a < act_dim:
                mask[a] = 1.0

        obs_t = torch.from_numpy(obs).to(device)
        mask_t = torch.from_numpy(mask).to(device)

        # Sample action from policy pi(a|s) for on-policy learning
        with torch.no_grad():
            logits, v = model(obs_t.unsqueeze(0))
            probs = masked_softmax(logits, mask_t.unsqueeze(0))

            # Safety: if mask is all-zero, fall back to uniform
            if torch.all(mask_t <= 0):
                probs = torch.full_like(probs, 1.0 / probs.shape[-1])

            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()
            logp = dist.log_prob(a)

        action = int(a.item())
        next_ts = env.step([action])

        # Reward for player p (typically only non-zero at terminal in tic_tac_toe)
        r = 0.0
        if hasattr(next_ts, "rewards") and 0 <= p < len(next_ts.rewards):
            r = float(next_ts.rewards[p])

        # Store transition
        obs_list.append(obs_t)
        mask_list.append(mask_t)
        act_list.append(torch.tensor(action, device=device, dtype=torch.long))
        logp_list.append(logp.squeeze(0))
        val_list.append(v.squeeze(0))
        rew_list.append(torch.tensor(r, device=device, dtype=torch.float32))
        done_list.append(torch.tensor(float(next_ts.last()), device=device, dtype=torch.float32))

        ts = next_ts
        steps += 1

    # Bootstrap value for last state (needed for GAE)
    with torch.no_grad():
        if ts.last():
            v_last = torch.zeros((), device=device)
        else:
            try:
                p_last = int(ts.observations["current_player"])
                obs_last = np.asarray(ts.observations["info_state"][p_last], dtype=np.float32).reshape(-1)
                obs_last_t = torch.from_numpy(obs_last).to(device)
                _, v_last_b = model(obs_last_t.unsqueeze(0))
                v_last = v_last_b.squeeze(0).squeeze(0)
            except Exception:
                v_last = torch.zeros((), device=device)

    # Stack lists into tensors
    obs = torch.stack(obs_list)
    mask = torch.stack(mask_list)
    act = torch.stack(act_list)
    logp_old = torch.stack(logp_list)
    val_old = torch.stack(val_list)
    rew = torch.stack(rew_list)
    done = torch.stack(done_list)

    # ----- GAE -----
    adv = torch.zeros_like(rew)
    last_gae = torch.zeros((), device=device)

    for t in reversed(range(ROLLOUT_STEPS)):
        nonterminal = 1.0 - done[t]
        v_next = v_last if t == ROLLOUT_STEPS - 1 else val_old[t + 1]
        delta = rew[t] + GAMMA * v_next * nonterminal - val_old[t]
        last_gae = delta + GAMMA * LAMBDA * nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + val_old
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return obs, act, mask, logp_old, ret, adv


def ppo_update(model: ActorCritic, optimizer, batch):
    """
    PPO clipped objective update.
    """
    obs, act, mask, logp_old, ret, adv = batch
    T = obs.shape[0]
    idxs = np.arange(T)

    for _ in range(EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, T, MINIBATCH):
            mb = idxs[start:start + MINIBATCH]

            logits, v = model(obs[mb])
            probs = masked_softmax(logits, mask[mb])

            # Safety: any rows with no legal actions -> uniform
            row_sums = mask[mb].sum(dim=-1, keepdim=True)
            bad = row_sums <= 0
            if bad.any():
                probs = torch.where(bad, torch.full_like(probs, 1.0 / probs.shape[-1]), probs)

            dist = torch.distributions.Categorical(probs=probs)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv[mb]
            pi_loss = -torch.mean(torch.min(surr1, surr2))

            vf_loss = 0.5 * torch.mean((v - ret[mb]) ** 2)
            ent = torch.mean(dist.entropy())

            loss = pi_loss + VF_COEF * vf_loss - ENT_COEF * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()