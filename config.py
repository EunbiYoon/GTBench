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
ENT_COEF = 0.001
LR = 1e-4
MAX_GRAD_NORM = 0.5

# Training schedule
ROLLOUT_STEPS = 512
NUM_UPDATES = 800
EPOCHS = 2
MINIBATCH = 256

# Opponent + evaluation
CFR_ITERS = 50
EVAL_EVERY_UPDATES = 50
EVAL_GAMES = 1000

# Logging / plotting
PRINT_EVERY_EPISODES = 5000
LOG_CSV = "ppo_log.csv"
PLOT_PNG = "ppo_learning_curve.png"