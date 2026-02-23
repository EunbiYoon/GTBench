
# 4x4 TicTacToe — PPO Self-Play

This project trains a PPO agent in a custom **4×4 TicTacToe (4-in-a-row)** environment using self-play.

Training logs include:
- **Reward**
- **Approximate Regret**

---

## ▶️ Run

### Train

```bash
python main.py
```

This will:

- Train the PPO agent
- Save the trained model to `policy.pt`
- Save example games as:
  - `policy_game_X.png` (policy plays as X / first player)
  - `policy_game_O.png` (policy plays as O / second player)
- Save training metrics to `train_log.csv`

---

### Plot Results

```bash
python plot.py --csv train_log.csv --out learning_curve.png
```

This generates learning curves (reward and regret) from the CSV log.

---

## 📁 Scripts

### `environment.py`
4×4 TicTacToe game logic and board rendering.

### `agents.py`
Neural network (Actor-Critic) with action masking.

### `train.py`
Collects self-play rollouts and performs PPO updates.

### `eval.py`
Evaluates model:
- reward (as X and O)
- approximate regret

### `main.py`
Runs training loop, performs evaluation, saves model and logs.

### `plot.py`
Generates reward/regret learning curves from `train_log.csv`.

---

## 📊 Output Files

### `policy.pt`
Saved PyTorch model weights (trained PPO policy).

### `train_log.csv`
Training log file containing:

- total steps
- update number
- number of episodes
- reward as X (`reward_p0`)
- reward as O (`reward_p1`)
- regret as X (`regret_p0`)
- regret as O (`regret_p1`)

Used for plotting learning curves.

### `policy_game_X.png`
One greedy self-play game where the trained policy plays as **X (first player)**.

### `policy_game_O.png`
One greedy self-play game where the trained policy plays as **O (second player)**.

---

## 🎮 Board Example

```
X_1 X_2  .   .
 .  O_1  .   .
 .   .   .   .
 .   .   .   .
```

Each move is labeled in order:
- `X_1`, `X_2`, ...
- `O_1`, `O_2`, ...

---

That's it.
