# config.py

# Environment
BOARD_N = 4
ACT_DIM = BOARD_N * BOARD_N
OBS_DIM = ACT_DIM + 1  # 16 cells + current player

# Training
SEED = 0
DEVICE = "cpu"  # "cuda" if available

NUM_UPDATES = 2000
ROLLOUT_STEPS = 2048

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
MAX_GRAD_NORM = 0.5

# Evaluation / logging
EVAL_EVERY_UPDATES = 50
EVAL_EPISODES = 200          # policy value MC eval episodes
REGRET_ROOT_SIMS = 200       # sims per action for root-only best-action search
REGRET_ROLLOUT_HORIZON = 16  # max plies to simulate (game ends <=16)

LOG_CSV = "train_log.csv"