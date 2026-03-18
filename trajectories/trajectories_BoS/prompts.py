"""
prompts.py — System and user prompt templates for P1 in Battle of the Sexes.

Three prompt types as specified in config:
  - nash_aware: nudges the model to reason about equilibria
  - opponent_modeling: forces explicit ToM reasoning (System Prompt A from §4)
  - baseline: no strategic guidance (System Prompt B from §4)
"""

from __future__ import annotations
from game import BoSConfig, Action


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

def system_prompt_nash_aware(cfg: BoSConfig) -> str:
    return f"""You are Player 1 in a repeated Battle of the Sexes game.

Payoff structure:
  - Both choose Opera:    you get {cfg.a}, opponent gets {cfg.b}
  - Both choose Football: you get {cfg.b}, opponent gets {cfg.a}
  - Mismatch:             both get {cfg.mismatch}

You prefer Opera (your payoff is {cfg.a} there vs {cfg.b} at Football).
Your opponent prefers Football.

Key facts you should reason about:
1. There are two pure-strategy Nash Equilibria: (Opera, Opera) and (Football, Football).
2. Miscoordination is the worst outcome for both players.
3. The mixed-strategy Nash Equilibrium has you play Opera with probability {cfg.a/(cfg.a+cfg.b):.3f}.
4. Your goal is to coordinate on a Nash Equilibrium while preferring the one that benefits you.

Think carefully about which equilibrium is most likely given the opponent's behavior so far,
and choose your action to maximize coordination and your own payoff."""


def system_prompt_opponent_modeling(cfg: BoSConfig) -> str:
    return f"""You are Player 1 in a repeated Battle of the Sexes game.

Payoff structure:
  - Both choose Opera:    you get {cfg.a}, opponent gets {cfg.b}
  - Both choose Football: you get {cfg.b}, opponent gets {cfg.a}
  - Mismatch:             both get {cfg.mismatch}

You prefer Opera. Your opponent prefers Football.

Before choosing your action each round, you MUST explicitly reason about:
1. What pattern do you observe in your opponent's past actions?
2. What TYPE of player is your opponent? (e.g. always one action, reactive, strategic)
3. What does your opponent THINK you will do next?
4. What is your opponent's best response to what they think you'll do?
5. Given all of the above, what is YOUR best action?

This multi-step reasoning is mandatory. Do not skip to a conclusion."""


def system_prompt_baseline(cfg: BoSConfig) -> str:
    return f"""You are Player 1 in a Battle of the Sexes game.

Payoff structure:
  - Both choose Opera:    you get {cfg.a}, opponent gets {cfg.b}
  - Both choose Football: you get {cfg.b}, opponent gets {cfg.a}
  - Mismatch:             both get {cfg.mismatch}

Choose the action that gives you the best payoff. State your action clearly."""


SYSTEM_PROMPTS = {
    "nash_aware": system_prompt_nash_aware,
    "opponent_modeling": system_prompt_opponent_modeling,
    "baseline": system_prompt_baseline,
}


# ---------------------------------------------------------------------------
# Per-round user prompt
# ---------------------------------------------------------------------------

def build_round_prompt(
    round_num: int,
    total_rounds: int,
    opponent_history: list[Action],
    p1_history: list[Action],
    p1_payoffs: list[float],
    opponent_type_hint: str | None = None,
) -> str:
    """
    Build the user message for a single round decision.

    opponent_type_hint: optionally reveal opponent type for ablation studies.
                        Leave None for standard (blind) generation.
    """
    lines = [f"Round {round_num} of {total_rounds}."]

    if p1_history:
        history_lines = []
        for i, (my_a, opp_a, pay) in enumerate(
            zip(p1_history, opponent_history, p1_payoffs), start=1
        ):
            history_lines.append(
                f"  Round {i}: You played {my_a}, Opponent played {opp_a} → your payoff: {pay}"
            )
        lines.append("\nHistory so far:")
        lines.extend(history_lines)
        lines.append(f"\nYour cumulative payoff so far: {sum(p1_payoffs):.1f}")
    else:
        lines.append("This is the first round — no history yet.")

    if opponent_type_hint:
        lines.append(f"\n[Ablation mode] Opponent type: {opponent_type_hint}")

    lines.append(
        "\nWhat is your action for this round? "
        "Reason step by step, then end with: Action: Opera  OR  Action: Football"
    )

    return "\n".join(lines)
