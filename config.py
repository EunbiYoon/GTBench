# config.py
# Centralized experiment configuration (edit only here).

GAME_NAME = "tic_tac_toe"
SEED = 0
DEVICE = "cpu"

# Model
HIDDEN = 64

# PPO / GAE
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
LR = 3e-4
MAX_GRAD_NORM = 0.5

# Training schedule (FAST)
ROLLOUT_STEPS = 128        # was 512
NUM_UPDATES = 30           # was 800
EPOCHS = 1                 # was 2
MINIBATCH = 128            # was 256

# Opponent + evaluation (FAST)
CFR_ITERS = 200            # was 50 (tictactoe는 200 정도면 충분히 강함)
EVAL_EVERY_UPDATES = 10    # was 50
EVAL_GAMES = 200           # was 1000 (MC eval이 시간 많이 잡아먹음)

# Logging / plotting
PRINT_EVERY_EPISODES = 1000
LOG_CSV = "ppo_log.csv"
PLOT_PNG = "ppo_learning_curve.png"

# Trajectory debug view
SHOW_TRAJ = True
SHOW_TRAJ_EVERY_EPISODES = 1000
MAX_TRAJ_STEPS_PRINT = 20