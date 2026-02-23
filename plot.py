# plot.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_from_csv(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)

    steps = df["steps"]

    plt.figure()
    if "reward_p0" in df.columns:
        plt.plot(steps, df["reward_p0"], label="Reward (seat X / P0)")
    if "reward_p1" in df.columns:
        plt.plot(steps, df["reward_p1"], label="Reward (seat O / P1)")
    if "regret_p0" in df.columns:
        plt.plot(steps, df["regret_p0"], label="Regret (root-only, seat X)")
    if "regret_p1" in df.columns:
        plt.plot(steps, df["regret_p1"], label="Regret (root-only, seat O)")

    plt.xlabel("Environment steps")
    plt.ylabel("Reward / Regret")
    plt.title("4x4 TicTacToe PPO: Reward & Regret")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="learning_curve.png")
    args = parser.parse_args()
    plot_from_csv(args.csv, args.out)