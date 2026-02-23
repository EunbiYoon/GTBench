# plot.py
# Automatically loads policy.pt and saves one game

import torch
import matplotlib.pyplot as plt

from environment import TicTacToe4x4Env
from agents import ActorCritic, masked_softmax
from config import OBS_DIM, ACT_DIM, DEVICE, PYTORCH_POLICY


def policy_action(model, obs, legal):
    obs_t = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
    mask = torch.zeros((1, ACT_DIM), device=DEVICE)
    mask[0, legal] = 1.0

    with torch.no_grad():
        logits, _ = model(obs_t)
        probs = masked_softmax(logits, mask)
        action = torch.argmax(probs, dim=-1).item()

    return action


def save_board(env):
    board_lines = env.render().split("\n")
    board = [line.split() for line in board_lines]

    fig, ax = plt.subplots()
    ax.axis("off")

    table = ax.table(cellText=board, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.savefig("plot/policy_game.png", bbox_inches="tight")
    plt.close(fig)


def main():
    model = ActorCritic(OBS_DIM, ACT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(PYTORCH_POLICY, map_location=DEVICE))
    model.eval()

    env = TicTacToe4x4Env()
    obs = env.reset(starting_player=1)

    while True:
        legal = env.legal_actions()
        if not legal:
            break

        action = policy_action(model, obs, legal)
        step = env.step(action)
        obs = step.observation

        if step.done:
            break

    save_board(env)
    print("🎮 Saved policy_game.png")


if __name__ == "__main__":
    main()