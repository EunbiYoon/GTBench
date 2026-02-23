# main.py
import csv

import torch
import torch.optim as optim

from environment import TicTacToe4x4Env
from agents import ActorCritic
from train import collect_rollout, ppo_update
from eval import eval_reward_and_regret
from config import (
    OBS_DIM,
    ACT_DIM,
    SEED,
    DEVICE,
    NUM_UPDATES,
    LR,
    ROLLOUT_STEPS,
    EVAL_EVERY_UPDATES,
    LOG_CSV,
)


def set_seed(seed: int):
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    env = TicTacToe4x4Env()

    model = ActorCritic(OBS_DIM, ACT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_counter = [0]
    total_steps = 0

    # CSV init
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["steps", "update", "episodes", "reward_p0", "reward_p1", "regret_p0", "regret_p1"])

    # initial eval
    r0, r1, g0, g1 = eval_reward_and_regret(TicTacToe4x4Env, model, DEVICE)
    print(f"[init] steps=0 R_P0={r0:.4f} R_P1={r1:.4f} Reg_P0={g0:.4f} Reg_P1={g1:.4f}")
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([0, 0, episode_counter[0], r0, r1, g0, g1])

    for upd in range(1, NUM_UPDATES + 1):
        batch = collect_rollout(
            env, model, ACT_DIM, DEVICE, episode_counter,
            debug_print_every_episode=0  # <- 원하면 100 같은 값으로, 100에피소드마다 보드 출력
        )
        total_steps += ROLLOUT_STEPS
        ppo_update(model, optimizer, batch)

        if upd % EVAL_EVERY_UPDATES == 0:
            r0, r1, g0, g1 = eval_reward_and_regret(TicTacToe4x4Env, model, DEVICE)
            print(
                f"[upd {upd:04d}] steps={total_steps} episodes={episode_counter[0]} "
                f"R_P0={r0:.4f} R_P1={r1:.4f} Reg_P0={g0:.4f} Reg_P1={g1:.4f}"
            )
            with open(LOG_CSV, "a", newline="") as f:
                csv.writer(f).writerow([total_steps, upd, episode_counter[0], r0, r1, g0, g1])

    print(f"Saved training log to {LOG_CSV}")
    print("Training done.")


if __name__ == "__main__":
    main()