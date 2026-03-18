"""
game.py — Battle of the Sexes game logic and equilibrium scoring.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["Opera", "Football"]
ACTIONS: list[Action] = ["Opera", "Football"]


@dataclass
class BoSConfig:
    a: float = 3.0        # P1's preferred payoff (both Opera)
    b: float = 2.0        # P2's preferred payoff (both Football)
    mismatch: float = 0.0
    rounds: int = 5

    @property
    def max_possible_payoff(self) -> float:
        return self.a * self.rounds

    def payoff(self, p1_action: Action, p2_action: Action) -> tuple[float, float]:
        if p1_action == "Opera" and p2_action == "Opera":
            return (self.a, self.b)
        elif p1_action == "Football" and p2_action == "Football":
            return (self.b, self.a)
        else:
            return (self.mismatch, self.mismatch)


@dataclass
class RoundResult:
    round_num: int
    p1_action: Action
    p2_action: Action
    p1_payoff: float
    p2_payoff: float
    p1_reasoning: str = ""
    is_coordinated: bool = False
    is_p1_preferred_ne: bool = False   # both chose Opera
    is_p2_preferred_ne: bool = False   # both chose Football


@dataclass
class Trajectory:
    opponent_type: str
    p1_system_prompt_type: str
    rounds: list[RoundResult] = field(default_factory=list)
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    @property
    def p1_total_payoff(self) -> float:
        return sum(r.p1_payoff for r in self.rounds)

    @property
    def p2_total_payoff(self) -> float:
        return sum(r.p2_payoff for r in self.rounds)

    @property
    def coordination_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(1 for r in self.rounds if r.is_coordinated) / len(self.rounds)

    @property
    def p1_preferred_ne_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(1 for r in self.rounds if r.is_p1_preferred_ne) / len(self.rounds)

    def to_dict(self) -> dict:
        return {
            "opponent_type": self.opponent_type,
            "p1_system_prompt_type": self.p1_system_prompt_type,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "p1_total_payoff": self.p1_total_payoff,
            "p2_total_payoff": self.p2_total_payoff,
            "coordination_rate": self.coordination_rate,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "p1_action": r.p1_action,
                    "p2_action": r.p2_action,
                    "p1_payoff": r.p1_payoff,
                    "p2_payoff": r.p2_payoff,
                    "p1_reasoning": r.p1_reasoning,
                    "is_coordinated": r.is_coordinated,
                    "is_p1_preferred_ne": r.is_p1_preferred_ne,
                    "is_p2_preferred_ne": r.is_p2_preferred_ne,
                }
                for r in self.rounds
            ],
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Keywords that signal different reasoning depths (from §6 of the PDF)
_NE_KEYWORDS = [
    "nash equilibrium", "nash", "equilibrium", "pure strategy", "mixed strategy",
    "dominant strategy", "best response",
]
_COORD_KEYWORDS = [
    "coordinate", "coordination", "focal point", "schelling", "common knowledge",
    "mutual", "both of us", "together",
]
_OPPONENT_KEYWORDS = [
    "opponent", "they will", "they are", "their strategy", "their type",
    "they prefer", "they expect", "what they", "predict", "infer",
    "theory of mind", "they think", "model my",
]
_TOM_KEYWORDS = [
    "they think i", "what my opponent thinks", "they expect me",
    "they believe i", "my opponent models", "second-order",
]


def _keyword_score(text: str, keywords: list[str]) -> float:
    """Returns fraction of keyword categories present (capped at 1.0)."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / max(len(keywords) * 0.3, 1), 1.0)


def score_trajectory(traj: Trajectory, cfg: BoSConfig, weights: dict) -> Trajectory:
    """
    Computes an equilibrium alignment score for the trajectory.

    Score components:
      1. ne_coordination: did P1 land on a pure NE each round?
      2. reasoning_depth: does the reasoning show NE / coordination / opponent modeling?
      3. payoff: normalized cumulative payoff for P1
    """
    # 1. NE coordination score
    ne_score = traj.coordination_rate  # 0..1

    # 2. Reasoning depth — aggregate across all rounds
    all_reasoning = " ".join(r.p1_reasoning for r in traj.rounds).lower()
    ne_kw = _keyword_score(all_reasoning, _NE_KEYWORDS)
    coord_kw = _keyword_score(all_reasoning, _COORD_KEYWORDS)
    opp_kw = _keyword_score(all_reasoning, _OPPONENT_KEYWORDS)
    tom_kw = _keyword_score(all_reasoning, _TOM_KEYWORDS)
    # Weighted sum — ToM gets a bonus (level 3 reasoning from §6.4)
    reasoning_score = 0.25 * ne_kw + 0.25 * coord_kw + 0.3 * opp_kw + 0.2 * tom_kw

    # 3. Payoff score — normalize against theoretical max
    max_payoff = cfg.max_possible_payoff
    payoff_score = traj.p1_total_payoff / max_payoff if max_payoff > 0 else 0.0

    w_ne = weights.get("ne_coordination_weight", 0.5)
    w_r = weights.get("reasoning_depth_weight", 0.3)
    w_p = weights.get("payoff_weight", 0.2)

    total = w_ne * ne_score + w_r * reasoning_score + w_p * payoff_score

    traj.score = round(total, 4)
    traj.score_breakdown = {
        "ne_coordination": round(ne_score, 4),
        "reasoning_depth": round(reasoning_score, 4),
        "ne_keywords": round(ne_kw, 4),
        "coord_keywords": round(coord_kw, 4),
        "opponent_keywords": round(opp_kw, 4),
        "tom_keywords": round(tom_kw, 4),
        "payoff_normalized": round(payoff_score, 4),
    }
    return traj


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Action | None:
    """Extract Opera/Football decision from model output."""
    text_lower = text.lower()

    # Look for explicit "Action: X" pattern first
    m = re.search(r"action\s*:\s*(opera|football)", text_lower)
    if m:
        return "Opera" if m.group(1) == "opera" else "Football"

    # Look for "I choose / I'll play / I play / my choice is"
    m = re.search(
        r"(?:i choose|i'll (?:choose|play|go with|pick)|i play|my (?:choice|action|decision) is)\s+(opera|football)",
        text_lower,
    )
    if m:
        return "Opera" if m.group(1) == "opera" else "Football"

    # Fallback: last mention of either action word
    positions = {}
    for action in ["opera", "football"]:
        idx = text_lower.rfind(action)
        if idx != -1:
            positions[action] = idx
    if positions:
        chosen = max(positions, key=positions.get)
        return "Opera" if chosen == "opera" else "Football"

    return None
