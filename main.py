# main.py
# PPO self-play training + periodic evaluation vs TinyLlama
# Prints progress at EVERY update.

import csv
import time

import numpy as np
import torch
import torch.optim as optim

from environment import TicTacToe4x4Env
from agents import ActorCritic
from train import collect_rollout, ppo_update
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

from eval import eval_reward_and_regret_vs_tinyllama


def set_seed(seed: int):
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

    print("🚀 Training started")
    start_time = time.time()

    for upd in range(1, NUM_UPDATES + 1):

        # -------- Rollout --------
        batch = collect_rollout(
            env=env,
            model=model,
            act_dim=ACT_DIM,
            device=DEVICE,
            episode_counter_ref=episode_counter,
)

        total_steps += ROLLOUT_STEPS

        # -------- PPO Update --------
        ppo_update(model, optimizer, batch)

        # -------- Progress Print (ALWAYS) --------
        elapsed = time.time() - start_time
        print(
            f"[UPDATE {upd:03d}/{NUM_UPDATES}] "
            f"steps={total_steps} "
            f"episodes={episode_counter[0]} "
            f"time={elapsed:.1f}s"
        )

        # -------- Periodic Evaluation --------
        if upd % EVAL_EVERY_UPDATES == 0:
            r0, r1, g0, g1 = eval_reward_and_regret_vs_tinyllama(
                env_cls=TicTacToe4x4Env,
                model=model,
                device=DEVICE,
                act_dim=ACT_DIM,
                episodes=5,            # fast eval
                sims_per_action=3,
                sims_baseline=5,
                ollama_model="tinyllama:latest",
                temperature=0.0,
                timeout_sec=10.0,
                policy_sample=False,
            )

            print(
                f"   📊 Eval → "
                f"R_P0={r0:.3f} R_P1={r1:.3f} "
                f"Reg_P0={g0:.3f} Reg_P1={g1:.3f}"
            )

            with open(LOG_CSV, "a", newline="") as f:
                csv.writer(f).writerow([total_steps, upd, episode_counter[0], r0, r1, g0, g1])

    print("✅ Training finished")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Saved log to {LOG_CSV}")


if __name__ == "__main__":
    main()