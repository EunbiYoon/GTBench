# main.py
# Training + evaluation + CSV logging (plotting is moved to plot.py)

import csv

import numpy as np
import pyspiel
import torch
import torch.optim as optim
from open_spiel.python import rl_environment

from agents import ActorCritic
from cfr_opponent import build_cfr_average_policy
from config import (
    CFR_ITERS,
    DEVICE,
    EVAL_EVERY_UPDATES,
    GAME_NAME,
    LOG_CSV,
    LR,
    NUM_UPDATES,
    ROLLOUT_STEPS,
    SEED,
)
from eval import eval_reward_and_regret_vs_fixed
from train import collect_rollout, ppo_update


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # ----------------------------
    # 1) Reproducibility
    # ----------------------------
    set_seed(SEED)

    # ----------------------------
    # 2) Training environment
    # ----------------------------
    env = rl_environment.Environment(GAME_NAME)
    env.seed(SEED)

    obs_dim = int(env.observation_spec()["info_state"][0])
    act_dim = int(env.action_spec()["num_actions"])

    # ----------------------------
    # 3) PPO model + optimizer
    # ----------------------------
    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ----------------------------
    # 4) Fixed CFR opponent
    # ----------------------------
    game = pyspiel.load_game(GAME_NAME)
    if game.num_players() != 2:
        raise ValueError("This script expects a 2-player game.")

    print(f"Building fixed CFR opponent: iters={CFR_ITERS} ...")
    cfr_policy = build_cfr_average_policy(game, CFR_ITERS)
    print("CFR ready.")

    # ----------------------------
    # 5) Logging setup
    # ----------------------------
    episode_counter = [0]
    total_steps = 0

    # Overwrite CSV each run
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["steps", "update", "episodes", "reward_p0", "reward_p1", "regret_p0", "regret_p1"]
        )

    # ----------------------------
    # 6) Initial evaluation
    # ----------------------------
    reward_p0, reward_p1, regret_p0, regret_p1 = eval_reward_and_regret_vs_fixed(
        GAME_NAME, model, cfr_policy, device=DEVICE
    )
    print(
        f"[init] steps=0 "
        f"R_P0={reward_p0:.4f} R_P1={reward_p1:.4f} "
        f"Reg_P0={regret_p0:.4f} Reg_P1={regret_p1:.4f}"
    )

    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([0, 0, episode_counter[0], reward_p0, reward_p1, regret_p0, regret_p1])

    # ----------------------------
    # 7) PPO training loop
    # ----------------------------
    for upd in range(1, NUM_UPDATES + 1):
        # collect on-policy rollouts via self-play
        batch = collect_rollout(env, model, act_dim, DEVICE, episode_counter)
        total_steps += ROLLOUT_STEPS

        # PPO update
        ppo_update(model, optimizer, batch)

        # periodic evaluation vs fixed CFR
        if upd % EVAL_EVERY_UPDATES == 0:
            reward_p0, reward_p1, regret_p0, regret_p1 = eval_reward_and_regret_vs_fixed(
                GAME_NAME, model, cfr_policy, device=DEVICE
            )
            print(
                f"[upd {upd:04d}] steps={total_steps} episodes={episode_counter[0]} "
                f"R_P0={reward_p0:.4f} R_P1={reward_p1:.4f} "
                f"Reg_P0={regret_p0:.4f} Reg_P1={regret_p1:.4f}"
            )

            with open(LOG_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [total_steps, upd, episode_counter[0], reward_p0, reward_p1, regret_p0, regret_p1]
                )

    print(f"Saved training log to {LOG_CSV}")
    print("Training done.")


if __name__ == "__main__":
    main()