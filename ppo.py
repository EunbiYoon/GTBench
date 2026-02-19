# ppo.py
# PPO (Clip + GAE + Entropy) for OpenSpiel RL environment (GTBench-compatible)
# Includes robust observation handling + periodic evaluation + learning-curve plot.
#
# Run:
#   python ppo.py --game_name kuhn_poker
#   python ppo.py --game_name tic_tac_toe --num_updates 50 --rollout_steps 512 --epochs 3
#   python ppo.py --game_name connect_four --device cpu --eval_every 10 --eval_episodes 20
#
# Notes:
# - Trains via self-play using ONE policy controlling all players.
# - Evaluation uses greedy (argmax) policy; reports win-rate for player 0 + avg return.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from open_spiel.python import rl_environment


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Robust observation helpers
# ----------------------------
def _get_obs_dict(ts) -> Dict[str, Any]:
    obs = getattr(ts, "observations", None)
    if not isinstance(obs, dict):
        raise ValueError("TimeStep has no observations dict. Use open_spiel.python.rl_environment.Environment.")
    return obs


def _get_current_player(ts) -> int:
    obs = _get_obs_dict(ts)
    if "current_player" in obs:
        return int(obs["current_player"])
    cp = getattr(ts, "current_player", None)
    if cp is None:
        raise KeyError("Cannot find current_player in timestep observations.")
    return int(cp)


def _get_info_state_for_player(ts, player_id: int) -> np.ndarray:
    obs = _get_obs_dict(ts)
    if "info_state" not in obs:
        raise KeyError("observations has no 'info_state'.")
    info_all = obs["info_state"]

    # Usually list/tuple per player. Sometimes a single vector.
    try:
        raw = info_all[player_id]
    except Exception:
        raw = info_all

    arr = np.asarray(raw, dtype=np.float32)
    return arr.reshape(-1)


def _get_legal_actions_for_player(ts, player_id: int) -> List[int]:
    obs = _get_obs_dict(ts)
    if "legal_actions" not in obs:
        raise KeyError("observations has no 'legal_actions'.")
    legal_all = obs["legal_actions"]

    # Common: list indexed by player_id -> list[int]
    try:
        legal = legal_all[player_id]
        if isinstance(legal, (list, tuple, np.ndarray)):
            return [int(a) for a in legal]
    except Exception:
        pass

    # Dict mapping player_id -> list[int]
    if isinstance(legal_all, dict) and player_id in legal_all:
        legal = legal_all[player_id]
        return [int(a) for a in legal]

    # Flat list (rare)
    if isinstance(legal_all, (list, tuple, np.ndarray)):
        return [int(a) for a in legal_all]

    raise ValueError(f"Unrecognized legal_actions structure: {type(legal_all)}")


def timestep_to_tensors(ts, device: str, act_dim: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    player_id = _get_current_player(ts)
    if player_id < 0:
        # Usually terminal/chance; rl_environment shouldn't request action here, but guard anyway.
        raise RuntimeError(f"Non-player timestep encountered (current_player={player_id}).")

    info_state = _get_info_state_for_player(ts, player_id)
    legal = _get_legal_actions_for_player(ts, player_id)

    mask = np.zeros((act_dim,), dtype=np.float32)
    for a in legal:
        if 0 <= a < act_dim:
            mask[a] = 1.0

    obs_t = torch.from_numpy(info_state).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    return obs_t, mask_t, player_id


# ----------------------------
# PPO Model
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)              # [B, A]
        value = self.critic(obs).squeeze(-1)  # [B]
        return logits, value


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # logits: [B, A], mask: [B, A] (0/1)
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))
    return torch.softmax(masked_logits, dim=dim)


def categorical_sample(probs: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(probs=probs).sample()


def categorical_logprob(probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(probs=probs).log_prob(actions)


# ----------------------------
# Config / Batch
# ----------------------------
@dataclass
class PPOConfig:
    game_name: str = "kuhn_poker"
    seed: int = 0
    device: str = "cpu"

    rollout_steps: int = 512
    num_updates: int = 50

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    epochs: int = 3
    minibatch_size: int = 256

    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    hidden: int = 128

    eval_every: int = 10
    eval_episodes: int = 30
    greedy_eval: bool = True
    save_curve_path: str = ""  # e.g., curve.npz


@dataclass
class TrajectoryBatch:
    obs: torch.Tensor
    act: torch.Tensor
    mask: torch.Tensor
    logp_old: torch.Tensor
    val_old: torch.Tensor
    ret: torch.Tensor
    adv: torch.Tensor


# ----------------------------
# Environment
# ----------------------------
def make_env(game_name: str, seed: int) -> rl_environment.Environment:
    env = rl_environment.Environment(game_name)
    env.seed(seed)
    return env


def get_obs_dim(env: rl_environment.Environment) -> int:
    spec = env.observation_spec()
    if "info_state" in spec:
        info = spec["info_state"]
        if isinstance(info, (list, tuple)) and len(info) > 0:
            return int(info[0])
        if isinstance(info, int):
            return int(info)
    raise KeyError(f"Unexpected observation_spec: {spec}")


def get_act_dim(env: rl_environment.Environment) -> int:
    spec = env.action_spec()
    if "num_actions" in spec:
        return int(spec["num_actions"])
    raise KeyError(f"Unexpected action_spec: {spec}")


# ----------------------------
# Rollout (self-play)
# ----------------------------
def collect_rollout(env: rl_environment.Environment, model: ActorCritic, cfg: PPOConfig) -> TrajectoryBatch:
    device = cfg.device
    act_dim = get_act_dim(env)

    obs_list, act_list, mask_list = [], [], []
    logp_list, val_list = [], []
    rew_list, done_list = [], []

    ts = env.reset()
    steps = 0

    while steps < cfg.rollout_steps:
        if ts.last():
            ts = env.reset()
            continue

        try:
            obs_t, mask_t, player_id = timestep_to_tensors(ts, device=device, act_dim=act_dim)
        except RuntimeError:
            ts = env.reset()
            continue

        obs_b = obs_t.unsqueeze(0)
        mask_b = mask_t.unsqueeze(0)

        with torch.no_grad():
            logits, v = model(obs_b)
            probs = masked_softmax(logits, mask_b)
            # if mask is all-zero (shouldn't), fallback uniform
            if torch.all(mask_b <= 0):
                probs = torch.full_like(probs, 1.0 / probs.shape[-1])

            a = categorical_sample(probs)  # [1]
            logp = categorical_logprob(probs, a)

        action = int(a.item())
        next_ts = env.step([action])

        # reward for acting player
        r = 0.0
        if hasattr(next_ts, "rewards") and isinstance(next_ts.rewards, (list, tuple, np.ndarray)):
            if 0 <= player_id < len(next_ts.rewards):
                r = float(next_ts.rewards[player_id])

        obs_list.append(obs_t)
        mask_list.append(mask_t)
        act_list.append(torch.tensor(action, device=device, dtype=torch.long))
        logp_list.append(logp.squeeze(0))
        val_list.append(v.squeeze(0))
        rew_list.append(torch.tensor(r, device=device, dtype=torch.float32))
        done_list.append(torch.tensor(float(next_ts.last()), device=device, dtype=torch.float32))

        ts = next_ts
        steps += 1

    obs = torch.stack(obs_list)
    mask = torch.stack(mask_list)
    act = torch.stack(act_list)
    logp_old = torch.stack(logp_list)
    val_old = torch.stack(val_list)
    rew = torch.stack(rew_list)
    done = torch.stack(done_list)

    # Bootstrap
    with torch.no_grad():
        if ts.last():
            v_last = torch.zeros((), device=device)
        else:
            try:
                obs_last, mask_last, _ = timestep_to_tensors(ts, device=device, act_dim=act_dim)
                _, v_last_b = model(obs_last.unsqueeze(0))
                v_last = v_last_b.squeeze(0).squeeze(0)
            except Exception:
                v_last = torch.zeros((), device=device)

    # GAE
    adv = torch.zeros_like(rew)
    last_gae = torch.zeros((), device=device)

    for t in reversed(range(cfg.rollout_steps)):
        nonterminal = 1.0 - done[t]
        v_next = v_last if t == cfg.rollout_steps - 1 else val_old[t + 1]
        delta = rew[t] + cfg.gamma * v_next * nonterminal - val_old[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + val_old
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return TrajectoryBatch(obs=obs, act=act, mask=mask, logp_old=logp_old, val_old=val_old, ret=ret, adv=adv)


# ----------------------------
# PPO Update
# ----------------------------
def ppo_update(model: ActorCritic, optimizer: optim.Optimizer, batch: TrajectoryBatch, cfg: PPOConfig) -> Dict[str, float]:
    T = batch.obs.shape[0]
    idxs = np.arange(T)

    pi_losses, vf_losses, entropies, total_losses, approx_kls = [], [], [], [], []

    for _ in range(cfg.epochs):
        np.random.shuffle(idxs)
        for start in range(0, T, cfg.minibatch_size):
            mb = idxs[start : start + cfg.minibatch_size]

            obs = batch.obs[mb]
            act = batch.act[mb]
            mask = batch.mask[mb]
            logp_old = batch.logp_old[mb]
            adv = batch.adv[mb]
            ret = batch.ret[mb]

            logits, v = model(obs)
            probs = masked_softmax(logits, mask)

            # Guard: any row with all-zero mask -> uniform
            row_sums = mask.sum(dim=-1, keepdim=True)
            bad = (row_sums <= 0)
            if bad.any():
                probs = torch.where(bad, torch.full_like(probs, 1.0 / probs.shape[-1]), probs)

            logp = categorical_logprob(probs, act)
            ratio = torch.exp(logp - logp_old)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            pi_loss = -torch.mean(torch.min(surr1, surr2))

            vf_loss = 0.5 * torch.mean((v - ret) ** 2)

            entropy = torch.mean(torch.distributions.Categorical(probs=probs).entropy())
            loss = pi_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean(logp_old - logp).item()

            pi_losses.append(pi_loss.item())
            vf_losses.append(vf_loss.item())
            entropies.append(entropy.item())
            total_losses.append(loss.item())
            approx_kls.append(approx_kl)

    return {
        "loss": float(np.mean(total_losses)),
        "pi_loss": float(np.mean(pi_losses)),
        "vf_loss": float(np.mean(vf_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
    }


# ----------------------------
# Evaluation + Learning curve
# ----------------------------
def run_one_episode(env: rl_environment.Environment, model: ActorCritic, device: str, greedy: bool = True) -> Tuple[float, int]:
    """
    Self-play one full episode using current policy.
    Returns:
      total_return_player0
      winner signal based on terminal reward of player0: 1 win, 0 draw, -1 loss
    """
    act_dim = get_act_dim(env)
    ts = env.reset()
    total_r0 = 0.0

    while not ts.last():
        player_id = _get_current_player(ts)
        if player_id < 0:
            break

        obs_t, mask_t, player_id = timestep_to_tensors(ts, device=device, act_dim=act_dim)
        with torch.no_grad():
            logits, _ = model(obs_t.unsqueeze(0))
            probs = masked_softmax(logits, mask_t.unsqueeze(0))
            if torch.all(mask_t <= 0):
                probs = torch.full_like(probs, 1.0 / probs.shape[-1])

            if greedy:
                action = int(torch.argmax(probs, dim=-1).item())
            else:
                action = int(categorical_sample(probs).item())

        ts = env.step([action])
        if hasattr(ts, "rewards") and len(ts.rewards) > 0:
            total_r0 += float(ts.rewards[0])

    winner = 0
    if hasattr(ts, "rewards") and len(ts.rewards) > 0:
        r0 = float(ts.rewards[0])
        if r0 > 0:
            winner = 1
        elif r0 < 0:
            winner = -1

    return total_r0, winner


def evaluate(env: rl_environment.Environment, model: ActorCritic, cfg: PPOConfig) -> Dict[str, float]:
    wins = draws = losses = 0
    rets = []

    for _ in range(cfg.eval_episodes):
        ep_ret, w = run_one_episode(env, model, device=cfg.device, greedy=cfg.greedy_eval)
        rets.append(ep_ret)
        if w == 1:
            wins += 1
        elif w == 0:
            draws += 1
        else:
            losses += 1

    return {
        "win_rate": wins / cfg.eval_episodes,
        "avg_return": float(np.mean(rets)),
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=str, default="kuhn_poker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--num_updates", type=int, default=50)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--stochastic_eval", action="store_true")  # if set, eval uses sampling instead of greedy
    parser.add_argument("--save_curve_path", type=str, default="") # e.g., curve.npz (saves steps/win/ret arrays)

    args = parser.parse_args()

    cfg = PPOConfig(
        game_name=args.game_name,
        seed=args.seed,
        device=args.device,
        num_updates=args.num_updates,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        hidden=args.hidden,
        lr=args.lr,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        greedy_eval=(not args.stochastic_eval),
        save_curve_path=args.save_curve_path,
    )

    set_seed(cfg.seed)

    env = make_env(cfg.game_name, cfg.seed)
    obs_dim = get_obs_dim(env)
    act_dim = get_act_dim(env)

    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=cfg.hidden).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history_steps: List[int] = []
    history_win: List[float] = []
    history_ret: List[float] = []

    total_env_steps = 0

    # initial eval
    ev0 = evaluate(env, model, cfg)
    history_steps.append(0)
    history_win.append(ev0["win_rate"])
    history_ret.append(ev0["avg_return"])
    print(f"[init] game={cfg.game_name} win={ev0['win_rate']:.2f} ret={ev0['avg_return']:.3f}")

    for upd in range(1, cfg.num_updates + 1):
        batch = collect_rollout(env, model, cfg)
        total_env_steps += cfg.rollout_steps

        metrics = ppo_update(model, optimizer, batch, cfg)

        if upd % cfg.eval_every == 0:
            ev = evaluate(env, model, cfg)
            history_steps.append(total_env_steps)
            history_win.append(ev["win_rate"])
            history_ret.append(ev["avg_return"])

            print(
                f"[{upd:04d}] steps={total_env_steps} "
                f"win={ev['win_rate']:.2f} ret={ev['avg_return']:.3f} "
                f"loss={metrics['loss']:.3f} ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f}"
            )

    # Save curve data (optional)
    if cfg.save_curve_path:
        np.savez(
            cfg.save_curve_path,
            steps=np.array(history_steps, dtype=np.int64),
            win=np.array(history_win, dtype=np.float32),
            ret=np.array(history_ret, dtype=np.float32),
            game=np.array([cfg.game_name]),
        )
        print(f"Saved learning curve to: {cfg.save_curve_path}")

    # Plot learning curve
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history_steps, history_win)
    plt.xlabel("Environment steps")
    plt.ylabel("Win rate (eval)")
    plt.title(f"PPO Learning Curve (Win Rate) - {cfg.game_name}")
    plt.show()

    plt.figure()
    plt.plot(history_steps, history_ret)
    plt.xlabel("Environment steps")
    plt.ylabel("Avg return (player 0, eval)")
    plt.title(f"PPO Learning Curve (Avg Return) - {cfg.game_name}")
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()