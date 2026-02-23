# 4x4 TicTacToe --- PPO Self-Play

This project trains a PPO agent in a custom **4×4 TicTacToe
(4-in-a-row)** environment using self-play.

Training logs: - **Reward** - **Approximate Regret**

------------------------------------------------------------------------

## ▶️ Run

### Train

``` bash
python main.py
```

### Plot Results

``` bash
python plot.py --csv train_log.csv --out learning_curve.png
```

------------------------------------------------------------------------

## 📁 Scripts

### `environment.py`

4×4 TicTacToe game logic and board rendering.

### `agents.py`

Neural network (Actor-Critic) + action masking.

### `train.py`

Collects self-play rollouts and performs PPO updates.

### `eval.py`

Evaluates model: - reward (as X and O) - approximate regret

### `main.py`

Runs training loop and saves logs.

### `plot.py`

Generates reward/regret learning curves from CSV.

------------------------------------------------------------------------

## 🎮 Board Example

    X_1 X_2  .   .
     .  O_1  .   .
     .   .   .   .
     .   .   .   .

------------------------------------------------------------------------

That's it.
