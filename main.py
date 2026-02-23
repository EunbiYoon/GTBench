# main.py
# PPO self-play training
# + periodic evaluation vs TinyLlama
# + save trained model
# + save one greedy policy game as image automatically

import csv
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from environment import TicTacToe4x4Env
from agents import ActorCritic, masked_softmax
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


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def policy_action(model, obs, legal):
    obs_t = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
    mask = torch.zeros((1, ACT_DIM), device=DEVICE)
    mask[0, legal] = 1.0

    with torch.no_grad():
        logits, _ = model(obs_t)
        probs = masked_softmax(logits, mask)
        action = torch.argmax(probs, dim=-1).item()

    return action


def save_policy_games(model):
    for start_player in [1, -1]:

        env = TicTacToe4x4Env()
        obs = env.reset(starting_player=start_player)

        while True:
            legal = env.legal_actions()
            if not legal:
                break

            action = policy_action(model, obs, legal)
            step = env.step(action)
            obs = step.observation

            if step.done:
                break

        board_lines = env.render().split("\n")
        board = [line.split() for line in board_lines]

        fig, ax = plt.subplots()
        ax.axis("off")

        table = ax.table(cellText=board, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        filename = "policy_game_X.png" if start_player == 1 else "policy_game_O.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)

        print(f"🎮 Saved {filename}")

# ----------------------------
# Main Training Loop
# ----------------------------

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

        batch = collect_rollout(
            env=env,
            model=model,
            act_dim=ACT_DIM,
            device=DEVICE,
            episode_counter_ref=episode_counter,
        )

        total_steps += ROLLOUT_STEPS
        ppo_update(model, optimizer, batch)

        elapsed = time.time() - start_time

        print(
            f"[{upd:03d}/{NUM_UPDATES}] "
            f"steps={total_steps} "
            f"episodes={episode_counter[0]} "
            f"time={elapsed:.1f}s"
        )

        # LLM Evaluation
        if upd % EVAL_EVERY_UPDATES == 0:
            print("   ⏳ Starting LLM evaluation...")
            r0, r1, g0, g1 = eval_reward_and_regret_vs_tinyllama(
                env_cls=TicTacToe4x4Env,
                model=model,
                device=DEVICE,
                act_dim=ACT_DIM,
                episodes=5,
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

    # Save model
    torch.save(model.state_dict(), "policy.pt")
    print("💾 Model saved to policy.pt")

    # Save one policy game visualization
    save_policy_games(model)

    print("✅ Training finished")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Saved log to {LOG_CSV}")


if __name__ == "__main__":
    main()