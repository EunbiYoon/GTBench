"""
Configuration for Battle of the Sexes DPO trajectory generation.
"""

# ── API Configuration ──────────────────────────────────────────────
LITELLM_BASE_URL = "https://thekeymaker.umass.edu/v1"
LITELLM_API_KEY = "sk-NraqXiinrjOk9Py7Fzp2FQ"

AGENT_MODEL = "gpt4o"
LLM_OPPONENT_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-haiku-4-5"

# ── Game Configuration ─────────────────────────────────────────────
ACTIONS = ["Opera", "Football"]

# Payoff matrix: PAYOFFS[player1_action][player2_action] = (p1_payoff, p2_payoff)
# Player 1 (agent) prefers Opera, Player 2 (opponent) prefers Football
PAYOFFS = {
    ("Opera", "Opera"): (3, 2),
    ("Opera", "Football"): (0, 0),
    ("Football", "Opera"): (0, 0),
    ("Football", "Football"): (2, 3),
}

NUM_ROUNDS = 5

# ── Opponent Configuration ─────────────────────────────────────────
EPSILON = 0.1  # noise probability for scripted opponents

OPPONENT_TYPES = [
    "AlwaysOpera",
    "AlwaysFootball",
    "Alternator",
    "Random",
    "ConditionalCooperator",
    "LLM",
]

# ── Scoring Configuration (for optional validation) ────────────────
# Weights for final score (must sum to 1.0)
WEIGHT_COORDINATION = 0.20
WEIGHT_PAYOFF = 0.10
WEIGHT_REASONING = 0.70

# For normalizing payoff to 0-100 scale
MAX_PAYOFF = NUM_ROUNDS * 3
MIN_PAYOFF = 0

# For normalizing coordination to 0-100 scale
MAX_COORDINATION = NUM_ROUNDS
MIN_COORDINATION = 0

# ── Data Generation Configuration ──────────────────────────────────
GAMES_PER_OPPONENT_TEST = 1
GAMES_PER_OPPONENT_PROD = 15

# ── Rate Limiting ──────────────────────────────────────────────────
REQUEST_DELAY_SECONDS = 1.0
