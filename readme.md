# 🎮 PPO vs Fixed CFR on Tic-Tac-Toe

This project implements a **Proximal Policy Optimization (PPO)** agent
trained via self-play on Tic-Tac-Toe using OpenSpiel.\
The trained PPO agent is periodically evaluated against a fixed **CFR
(Counterfactual Regret Minimization)** opponent using Monte-Carlo
rollouts.

------------------------------------------------------------------------

## 🚀 Overview

-   🤖 **Training:** PPO with self-play (shared network for both
    players)
-   🎯 **Opponent:** Fixed CFR average policy
-   📊 **Evaluation:** Monte-Carlo expected value (both seatings)
-   📁 **Logging:** CSV logs + learning curve plot

------------------------------------------------------------------------

## 📂 Project Structure

    gtbench/
    │
    ├── config.py          # ⚙️ Hyperparameters and experiment configuration
    ├── agents.py          # 🧠 Actor-Critic model + action sampling utilities
    ├── train.py           # 🔄 Rollout collection + PPO update logic
    ├── cfr_opponent.py    # 🧮 CFR training + action sampling
    ├── eval.py            # 📈 Monte-Carlo evaluation vs fixed CFR
    ├── ppo.py             # ▶️ Main training script
    └── README.md

------------------------------------------------------------------------

## 🏋️ How It Works

### 1️⃣ PPO Training (Self-Play)

-   A single Actor-Critic network controls **both Player 0 and Player
    1**
-   Rollouts are collected on-policy
-   PPO clipped objective is used for updates
-   GAE (Generalized Advantage Estimation) stabilizes training

The model improves by playing against itself 🎲.

------------------------------------------------------------------------

### 2️⃣ Fixed CFR Opponent

Before PPO training begins:

-   A CFR solver runs for a fixed number of iterations
-   The average CFR policy is extracted
-   This policy remains fixed during PPO training

This provides a strong equilibrium-style baseline opponent 🧩.

------------------------------------------------------------------------

### 3️⃣ Evaluation

Evaluation is performed periodically:

-   PPO as Player 0 vs CFR as Player 1
-   PPO as Player 1 vs CFR as Player 0

Monte-Carlo rollouts estimate:

-   📌 **EV_P0**: Expected reward when PPO plays first
-   📌 **EV_P1**: Expected reward when PPO plays second

------------------------------------------------------------------------

## ▶️ Running the Experiment

From the project root directory:

``` bash
python ppo.py
```

The script will:

1.  🧮 Build the fixed CFR opponent\
2.  🔄 Train PPO via self-play\
3.  📊 Periodically evaluate against CFR\
4.  💾 Save:
    -   `ppo_log.csv`
    -   `ppo_learning_curve.png`

------------------------------------------------------------------------

## ⚙️ Key Hyperparameters

Edit `config.py` to modify:

-   📉 Learning rate
-   🔁 Rollout steps
-   🔄 Number of updates
-   🎲 Entropy coefficient
-   🧮 CFR iterations
-   📊 Evaluation frequency

------------------------------------------------------------------------

## 📦 Requirements

-   Python 3.8+
-   PyTorch
-   OpenSpiel
-   NumPy
-   Matplotlib

Example installation:

``` bash
pip install torch numpy matplotlib
pip install open_spiel
```

------------------------------------------------------------------------

## 📝 Notes

-   PPO uses **stochastic action sampling**, not greedy actions.
-   Tic-Tac-Toe is highly sensitive to small mistakes ⚠️.
-   Training is fully self-play; evaluation uses a stronger fixed
    opponent.

------------------------------------------------------------------------

✨ Happy experimenting!
