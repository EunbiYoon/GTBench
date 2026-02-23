# config.py (BALANCED FAST VERSION)

# Environment
BOARD_N = 4
ACT_DIM = BOARD_N * BOARD_N
OBS_DIM = ACT_DIM + 1

# Training
SEED = 0
DEVICE = "cpu"

NUM_UPDATES = 40         # 50 → 40 (더 빠르게)
ROLLOUT_STEPS = 512      # 충분히 빠름

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

PPO_EPOCHS = 2
MINIBATCH_SIZE = 128
MAX_GRAD_NORM = 0.5

# Evaluation / logging
EVAL_EVERY_UPDATES = 5     # 🔥 1 → 5 (LLM 호출 줄이기)
EVAL_EPISODES = 5          # 🔥 10 → 5 (더 빠름)
REGRET_ROOT_SIMS = 3       # 🔥 5 → 3
REGRET_ROLLOUT_HORIZON = 16

LOG_CSV = "log/train_log.csv"