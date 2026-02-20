# ppo.py
# Tic-Tac-Toe PPO (self-play) + Fixed CFR opponent eval (EV + NashConv)
# - No CLI args: just run `python ppo.py`
# - Prints progress every 1000 completed episodes
# - Fast config at top

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr as cfr_lib
from open_spiel.python.algorithms import exploitability as expl_lib
from open_spiel.python import policy as policy_lib


# =========================
# FAST CONFIG (edit only here)
# =========================
GAME_NAME = "tic_tac_toe"
SEED = 0
DEVICE = "cpu"

ROLLOUT_STEPS = 256          # smaller => faster
NUM_UPDATES = 40             # smaller => faster
EPOCHS = 2
MINIBATCH = 128

HIDDEN = 64
LR = 3e-4

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5

PRINT_EVERY_EPISODES = 1000  # progress print
EVAL_EVERY_UPDATES = 10      # eval print
CFR_ITERS = 1000             # smaller => faster
# =========================


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------
# Model
# ----------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))
    return torch.softmax(masked_logits, dim=-1)


# ----------------------
# Rollout (self-play) + episode progress
# ----------------------
def collect_rollout(env, model, act_dim: int, device: str, episode_counter_ref):
    obs_list, act_list, mask_list = [], [], []
    logp_list, val_list = [], []
    rew_list, done_list = [], []

    ts = env.reset()
    steps = 0

    while steps < ROLLOUT_STEPS:
        if ts.last():
            episode_counter_ref[0] += 1
            if episode_counter_ref[0] % PRINT_EVERY_EPISODES == 0:
                print(f"[progress] episodes={episode_counter_ref[0]}")
            ts = env.reset()
            continue

        p = int(ts.observations["current_player"])
        if p < 0:
            ts = env.reset()
            continue

        obs = np.asarray(ts.observations["info_state"][p], dtype=np.float32).reshape(-1)
        legal = ts.observations["legal_actions"][p]

        mask = np.zeros((act_dim,), dtype=np.float32)
        for a in legal:
            if 0 <= a < act_dim:
                mask[a] = 1.0

        obs_t = torch.from_numpy(obs).to(device)
        mask_t = torch.from_numpy(mask).to(device)

        with torch.no_grad():
            logits, v = model(obs_t.unsqueeze(0))
            probs = masked_softmax(logits, mask_t.unsqueeze(0))
            if torch.all(mask_t <= 0):
                probs = torch.full_like(probs, 1.0 / probs.shape[-1])
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()               # [1]
            logp = dist.log_prob(a)         # [1]

        action = int(a.item())
        next_ts = env.step([action])

        r = 0.0
        if hasattr(next_ts, "rewards") and 0 <= p < len(next_ts.rewards):
            r = float(next_ts.rewards[p])

        obs_list.append(obs_t)
        mask_list.append(mask_t)
        act_list.append(torch.tensor(action, device=device, dtype=torch.long))
        logp_list.append(logp.squeeze(0))
        val_list.append(v.squeeze(0))
        rew_list.append(torch.tensor(r, device=device, dtype=torch.float32))
        done_list.append(torch.tensor(float(next_ts.last()), device=device, dtype=torch.float32))

        ts = next_ts
        steps += 1

    # Bootstrap value for last state
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

    obs = torch.stack(obs_list)
    mask = torch.stack(mask_list)
    act = torch.stack(act_list)
    logp_old = torch.stack(logp_list)
    val_old = torch.stack(val_list)
    rew = torch.stack(rew_list)
    done = torch.stack(done_list)

    # GAE
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


# ----------------------
# PPO Update
# ----------------------
def ppo_update(model, optimizer, batch):
    obs, act, mask, logp_old, ret, adv = batch
    T = obs.shape[0]
    idxs = np.arange(T)

    for _ in range(EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, T, MINIBATCH):
            mb = idxs[start : start + MINIBATCH]

            logits, v = model(obs[mb])
            probs = masked_softmax(logits, mask[mb])

            # guard any all-zero mask rows
            row_sums = mask[mb].sum(dim=-1, keepdim=True)
            bad = (row_sums <= 0)
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


# ----------------------
# CFR fixed opponent
# ----------------------
def build_cfr_average_policy(game, iters: int):
    solver = cfr_lib.CFRSolver(game)
    for _ in range(iters):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


# ----------------------
# Torch policy wrapper for exact EV + NashConv
# ----------------------
class TorchJointPolicy(policy_lib.Policy):
    def __init__(self, game, model, device: str):
        super().__init__(game)
        self._game = game
        self._model = model
        self._device = device
        self._num_actions = game.num_distinct_actions()

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()

        legal = state.legal_actions(player_id)
        if not legal:
            return {}

        obs = np.asarray(state.information_state_tensor(player_id), dtype=np.float32).reshape(-1)
        obs_t = torch.from_numpy(obs).to(self._device).unsqueeze(0)

        mask = np.zeros((self._num_actions,), dtype=np.float32)
        for a in legal:
            if 0 <= a < self._num_actions:
                mask[a] = 1.0
        mask_t = torch.from_numpy(mask).to(self._device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self._model(obs_t)
            probs = masked_softmax(logits, mask_t).squeeze(0)

        out = {a: float(probs[a].item()) for a in legal}
        s = sum(out.values())
        if s <= 0:
            uni = 1.0 / len(legal)
            return {a: uni for a in legal}
        if abs(s - 1.0) > 1e-6:
            for a in out:
                out[a] /= s
        return out


def expected_value_player0(game, policies):
    # exact tree traversal EV for player0
    def rec(state):
        if state.is_terminal():
            return float(state.returns()[0])
        if state.is_chance_node():
            ev = 0.0
            for a, p in state.chance_outcomes():
                nxt = state.clone()
                nxt.apply_action(a)
                ev += p * rec(nxt)
            return ev

        p = state.current_player()
        ap = policies[p].action_probabilities(state, p)
        ev = 0.0
        for a, pa in ap.items():
            if pa <= 0:
                continue
            nxt = state.clone()
            nxt.apply_action(a)
            ev += pa * rec(nxt)
        return ev

    return rec(game.new_initial_state())


def eval_vs_fixed_cfr(game, ppo_policy, cfr_policy):
    # PPO as P0 vs CFR as P1
    ev_p0 = expected_value_player0(game, [ppo_policy, cfr_policy])
    # CFR as P0 vs PPO as P1 (still returns player0 EV)
    ev_p1 = expected_value_player0(game, [cfr_policy, ppo_policy])
    # PPO's EV when playing as P1 is the negative of player0 EV in the second seating (zero-sum)
    return float(ev_p0), float(-ev_p1)


# ----------------------
# Main
# ----------------------
def main():
    set_seed(SEED)

    # training env
    env = rl_environment.Environment(GAME_NAME)
    env.seed(SEED)

    obs_dim = int(env.observation_spec()["info_state"][0])
    act_dim = int(env.action_spec()["num_actions"])

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # eval game (exact tree)
    game = pyspiel.load_game(GAME_NAME)
    if game.num_players() != 2:
        raise ValueError("This script expects a 2-player game.")

    print(f"Building fixed CFR opponent: iters={CFR_ITERS} ...")
    cfr_policy = build_cfr_average_policy(game, CFR_ITERS)
    print("CFR ready.")

    ppo_policy = TorchJointPolicy(game, model, DEVICE)

    episode_counter = [0]
    total_steps = 0

    # initial eval
    ev_as_p0, ev_as_p1 = eval_vs_fixed_cfr(game, ppo_policy, cfr_policy)
    nc = float(expl_lib.nash_conv(game, ppo_policy))
    print(f"[init] steps=0 EV(PPO as P0 vs CFR)={ev_as_p0:.4f} EV(PPO as P1 vs CFR)={ev_as_p1:.4f} NashConv={nc:.4f}")

    for upd in range(1, NUM_UPDATES + 1):
        batch = collect_rollout(env, model, act_dim, DEVICE, episode_counter)
        total_steps += ROLLOUT_STEPS
        ppo_update(model, optimizer, batch)

        if upd % EVAL_EVERY_UPDATES == 0:
            ev_as_p0, ev_as_p1 = eval_vs_fixed_cfr(game, ppo_policy, cfr_policy)
            nc = float(expl_lib.nash_conv(game, ppo_policy))
            print(
                f"[upd {upd:03d}] steps={total_steps} episodes={episode_counter[0]} "
                f"EV_P0={ev_as_p0:.4f} EV_P1={ev_as_p1:.4f} NashConv={nc:.4f}"
            )

    print("Done.")


if __name__ == "__main__":
    main()