# ppo.py
# Entry point: orchestrates training + evaluation + logging + plotting.

import csv
import numpy as np
import torch
import torch.optim as optim
import pyspiel
import matplotlib.pyplot as plt

from open_spiel.python import rl_environment

from config import (
    GAME_NAME, SEED, DEVICE,
    LR, CFR_ITERS, EVAL_GAMES, EVAL_EVERY_UPDATES,
    ROLLOUT_STEPS, NUM_UPDATES,
    LOG_CSV, PLOT_PNG
)
from agents import ActorCritic
from train import collect_rollout, ppo_update
from cfr_opponent import build_cfr_average_policy
from eval import eval_mc_vs_fixed


def set_seed(seed: int):
    """Seed helper (kept here to avoid circular imports)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # -------------------------------------------------
    # 1) Reproducibility
    # -------------------------------------------------
    set_seed(SEED)

    # -------------------------------------------------
    # 2) Training environment (self-play PPO)
    # -------------------------------------------------
    env = rl_environment.Environment(GAME_NAME)
    env.seed(SEED)

    # Read observation/action space sizes from environment specs
    obs_dim = int(env.observation_spec()["info_state"][0])
    act_dim = int(env.action_spec()["num_actions"])

    # -------------------------------------------------
    # 3) Initialize PPO model + optimizer
    # -------------------------------------------------
    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------------------------------------------------
    # 4) Build fixed CFR opponent policy (strong baseline)
    # -------------------------------------------------
    game = pyspiel.load_game(GAME_NAME)
    if game.num_players() != 2:
        raise ValueError("This script expects a 2-player game.")

    print(f"Building fixed CFR opponent: iters={CFR_ITERS} ...")
    cfr_policy = build_cfr_average_policy(game, CFR_ITERS)
    print("CFR ready.")

    # -------------------------------------------------
    # 5) Logging state
    # -------------------------------------------------
    episode_counter = [0]  # updated inside collect_rollout
    total_steps = 0

    history_steps, history_ev_p0, history_ev_p1 = [], [], []

    # Initialize CSV log file (overwrite each run)
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["steps", "update", "episodes", "ev_p0", "ev_p1"])

    # -------------------------------------------------
    # 6) Initial evaluation before training (baseline)
    # -------------------------------------------------
    ev_p0, ev_p1 = eval_mc_vs_fixed(GAME_NAME, model, cfr_policy, n_games=EVAL_GAMES, seed=SEED, device=DEVICE)
    print(f"[init] steps=0 EV_P0={ev_p0:.4f} EV_P1={ev_p1:.4f}")

    history_steps.append(0)
    history_ev_p0.append(ev_p0)
    history_ev_p1.append(ev_p1)

    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([0, 0, episode_counter[0], ev_p0, ev_p1])

    # -------------------------------------------------
    # 7) PPO training loop
    # -------------------------------------------------
    for upd in range(1, NUM_UPDATES + 1):
        # (a) Collect on-policy trajectories via self-play
        batch = collect_rollout(env, model, act_dim, DEVICE, episode_counter)
        total_steps += ROLLOUT_STEPS

        # (b) Optimize policy/value with PPO
        ppo_update(model, optimizer, batch)

        # (c) Periodic evaluation (MC) vs fixed CFR opponent
        if upd % EVAL_EVERY_UPDATES == 0:
            ev_p0, ev_p1 = eval_mc_vs_fixed(
                GAME_NAME,
                model,
                cfr_policy,
                n_games=EVAL_GAMES,
                seed=SEED + upd,  # vary seed slightly across evals
                device=DEVICE,
            )
            print(f"[upd {upd:04d}] steps={total_steps} episodes={episode_counter[0]} EV_P0={ev_p0:.4f} EV_P1={ev_p1:.4f}")

            history_steps.append(total_steps)
            history_ev_p0.append(ev_p0)
            history_ev_p1.append(ev_p1)

            with open(LOG_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([total_steps, upd, episode_counter[0], ev_p0, ev_p1])

    # -------------------------------------------------
    # 8) Plot learning curve
    # -------------------------------------------------
    plt.figure()
    plt.plot(history_steps, history_ev_p0, label="EV PPO as P0 vs CFR")
    plt.plot(history_steps, history_ev_p1, label="CFR vs EV PPO as P1")
    plt.xlabel("Environment steps")
    plt.ylabel("Expected Value (MC)")
    plt.title("PPO Learning Curve vs Fixed CFR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=150)

    print(f"Saved: {PLOT_PNG} and {LOG_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()