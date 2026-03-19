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

# ── Scoring Configuration ──────────────────────────────────────────
# Weights for final score (must sum to 1.0)
WEIGHT_COORDINATION = 0.20
WEIGHT_PAYOFF = 0.10
WEIGHT_REASONING = 0.70

# For normalizing payoff to 0-100 scale
# Max possible payoff in 5 rounds: 3 * 5 = 15 (coordinate on Opera every round)
# Min possible payoff: 0 (miscoordinate every round)
MAX_PAYOFF = NUM_ROUNDS * 3
MIN_PAYOFF = 0

# For normalizing coordination to 0-100 scale
MAX_COORDINATION = NUM_ROUNDS  # coordinate every round
MIN_COORDINATION = 0

# ── Data Generation Configuration ──────────────────────────────────
TRAJECTORIES_PER_OPPONENT_TEST = 2
TRAJECTORIES_PER_OPPONENT_PROD = 50

# Top/bottom percentile for labeling preferred/non-preferred
LABEL_TOP_PERCENTILE = 25  # top 25% = preferred
LABEL_BOTTOM_PERCENTILE = 25  # bottom 25% = non-preferred

# ── Rate Limiting ──────────────────────────────────────────────────
REQUEST_DELAY_SECONDS = 1.0  # delay between API calls to avoid rate limits
