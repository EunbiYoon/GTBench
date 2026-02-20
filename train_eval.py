# train_eval.py
# Rollout collection, PPO update, CFR opponent, and Monte-Carlo evaluation.

import numpy as np
import torch
import torch.nn as nn
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr as cfr_lib

from config import (
    ROLLOUT_STEPS, PRINT_EVERY_EPISODES,
    GAMMA, LAMBDA, CLIP_EPS, VF_COEF, ENT_COEF, MAX_GRAD_NORM,
    EPOCHS, MINIBATCH
)
from agents import masked_softmax, ActorCritic, sample_action_from_env_timestep


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
        # If episode ended, count it and reset environment
        if ts.last():
            episode_counter_ref[0] += 1
            if episode_counter_ref[0] % PRINT_EVERY_EPISODES == 0:
                print(f"[progress] episodes={episode_counter_ref[0]}")
            ts = env.reset()
            continue

        # Current player id (0 or 1 for tic_tac_toe)
        p = int(ts.observations["current_player"])
        if p < 0:
            ts = env.reset()
            continue

        # Build observation + legal mask for current player
        obs = np.asarray(ts.observations["info_state"][p], dtype=np.float32).reshape(-1)
        legal = ts.observations["legal_actions"][p]

        mask = np.zeros((act_dim,), dtype=np.float32)
        for a in legal:
            if 0 <= a < act_dim:
                mask[a] = 1.0

        obs_t = torch.from_numpy(obs).to(device)
        mask_t = torch.from_numpy(mask).to(device)

        # Sample action from current policy (stochastic policy gradient)
        with torch.no_grad():
            logits, v = model(obs_t.unsqueeze(0))  # [1, A], [1]
            probs = masked_softmax(logits, mask_t.unsqueeze(0))

            # Guard: if mask is all-zero (shouldn't happen, but keep safe)
            if torch.all(mask_t <= 0):
                probs = torch.full_like(probs, 1.0 / probs.shape[-1])

            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()              # [1]
            logp = dist.log_prob(a)        # [1]

        action = int(a.item())
        next_ts = env.step([action])

        # Reward for current player (zero-sum: usually only non-zero at terminal)
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
    obs = torch.stack(obs_list)        # [T, obs_dim]
    mask = torch.stack(mask_list)      # [T, act_dim]
    act = torch.stack(act_list)        # [T]
    logp_old = torch.stack(logp_list)  # [T]
    val_old = torch.stack(val_list)    # [T]
    rew = torch.stack(rew_list)        # [T]
    done = torch.stack(done_list)      # [T]

    # ----- GAE (Generalized Advantage Estimation) -----
    adv = torch.zeros_like(rew)
    last_gae = torch.zeros((), device=device)

    for t in reversed(range(ROLLOUT_STEPS)):
        nonterminal = 1.0 - done[t]
        v_next = v_last if t == ROLLOUT_STEPS - 1 else val_old[t + 1]
        delta = rew[t] + GAMMA * v_next * nonterminal - val_old[t]
        last_gae = delta + GAMMA * LAMBDA * nonterminal * last_gae
        adv[t] = last_gae

    # Return target for critic
    ret = adv + val_old

    # Normalize advantages for stability
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return obs, act, mask, logp_old, ret, adv


def ppo_update(model: ActorCritic, optimizer, batch):
    """
    Standard PPO clipped objective update.

    Uses:
      - ratio = exp(logp_new - logp_old)
      - clipped surrogate objective
      - value loss + entropy regularization
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

            # Guard: any rows with no legal actions -> uniform
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


def build_cfr_average_policy(game, iters: int):
    """
    Train a CFR solver for `iters` iterations and return the average policy.
    This serves as a strong fixed opponent (approximate equilibrium).
    """
    solver = cfr_lib.CFRSolver(game)
    for _ in range(iters):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def cfr_action_from_state(state, player_id: int, cfr_policy) -> int:
    """
    Sample an action from CFR policy for a pyspiel.State.
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


def eval_mc_vs_fixed(game_name: str, model: ActorCritic, cfr_policy, n_games: int, seed: int, device: str):
    """
    Monte-Carlo evaluation of PPO vs fixed CFR, in both seatings.

    We keep evaluation consistent with training by:
      - PPO actions sampled from rl_environment info_state
      - CFR actions sampled from synchronized pyspiel.State

    Returns:
      (ev_p0, ev_p1)
      - ev_p0: PPO reward when PPO plays as player 0 vs CFR as player 1
      - ev_p1: PPO reward when PPO plays as player 1 vs CFR as player 0
    """
    env = rl_environment.Environment(game_name)
    env.seed(seed)
    game = pyspiel.load_game(game_name)

    act_dim = int(env.action_spec()["num_actions"])

    # ---------- Seating 1: PPO as player 0 ----------
    ev_p0 = 0.0
    for _ in range(n_games):
        ts = env.reset()
        st = game.new_initial_state()

        while not ts.last():
            p = int(ts.observations["current_player"])
            if p < 0:
                ts = env.reset()
                st = game.new_initial_state()
                continue

            if p == 0:
                a = sample_action_from_env_timestep(ts, 0, model, act_dim, device)
            else:
                a = cfr_action_from_state(st, 1, cfr_policy)

            ts = env.step([a])
            st.apply_action(a)

        ev_p0 += float(ts.rewards[0])
    ev_p0 /= n_games

    # ---------- Seating 2: PPO as player 1 ----------
    ev_p1 = 0.0
    for _ in range(n_games):
        ts = env.reset()
        st = game.new_initial_state()

        while not ts.last():
            p = int(ts.observations["current_player"])
            if p < 0:
                ts = env.reset()
                st = game.new_initial_state()
                continue

            if p == 0:
                a = cfr_action_from_state(st, 0, cfr_policy)
            else:
                a = sample_action_from_env_timestep(ts, 1, model, act_dim, device)

            ts = env.step([a])
            st.apply_action(a)

        ev_p1 += float(ts.rewards[1])
    ev_p1 /= n_games

    return float(ev_p0), float(ev_p1)