"""
Game engine for Battle of the Sexes DPO data generation.

Pipeline per game:
1. Level_2 agent plays a full live game against an opponent → canonical history
2. At each round, level_0 agent responds to the same history (hypothetical)
3. DPO pairs are formed: level_2 = chosen, level_0 = rejected
"""

from config import PAYOFFS, NUM_ROUNDS
from agent import Agent, build_dpo_prompt


def run_game_and_collect_dpo(opponent, game_id: str = "0") -> dict:
    """
    Run a single game and produce per-round DPO pairs.

    Steps:
      1. Level_2 agent plays live against the opponent for NUM_ROUNDS
      2. For each round, level_0 responds to the history up to that point
      3. Each round produces one DPO pair

    Returns:
        {
            "game_id": str,
            "opponent_type": str,
            "canonical_history": [...],  # full game history from level_2
            "rounds": [
                {
                    "round_num": int,
                    "history_before": [...],  # history visible at decision time
                    "dpo_prompt": str,        # XML-formatted shared prompt
                    "chosen": {               # level_2 response
                        "reasoning": str,
                        "action": str,
                        "raw_response": str,
                    },
                    "rejected": {             # level_0 response
                        "reasoning": str,
                        "action": str,
                        "raw_response": str,
                    },
                    "canonical_action": str,  # what level_2 actually played
                    "opponent_action": str,
                    "payoff": (int, int),
                },
                ...
            ],
            "total_agent_payoff": int,
            "coordination_count": int,
        }
    """
    level_2_agent = Agent(reasoning_level="level_2")
    level_0_agent = Agent(reasoning_level="level_0")

    history = []  # canonical history built from level_2's live game
    rounds_data = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"    Round {round_num}/{NUM_ROUNDS}...")

        # History up to this point (before this round's actions)
        history_before = list(history)

        # ── Level_2 plays live ─────────────────────────────────────
        level_2_response = level_2_agent.get_response(history_before, round_num)
        agent_action = level_2_response["action"]

        # ── Opponent responds (live, reacts to level_2's history) ──
        opponent_action = opponent.choose_action(history_before)

        # ── Level_0 responds to the SAME history (hypothetical) ────
        level_0_response = level_0_agent.get_response(history_before, round_num)

        # ── Build the DPO prompt for this round ───────────────────
        dpo_prompt = build_dpo_prompt(history_before, round_num)

        # ── Compute payoff (based on level_2's actual play) ───────
        payoff = PAYOFFS[(agent_action, opponent_action)]

        # ── Record round data ─────────────────────────────────────
        round_data = {
            "round_num": round_num,
            "history_before": history_before,
            "dpo_prompt": dpo_prompt,
            "chosen": {
                "reasoning": level_2_response["reasoning"],
                "action": level_2_response["action"],
                "raw_response": level_2_response["raw_response"],
            },
            "rejected": {
                "reasoning": level_0_response["reasoning"],
                "action": level_0_response["action"],
                "raw_response": level_0_response["raw_response"],
            },
            "canonical_action": agent_action,
            "opponent_action": opponent_action,
            "payoff": payoff,
        }
        rounds_data.append(round_data)

        # ── Update canonical history with level_2's action ────────
        history.append({
            "agent": agent_action,
            "opponent": opponent_action,
        })

        print(
            f"      Level_2: {agent_action}, Level_0: {level_0_response['action']}, "
            f"Opponent: {opponent_action} → Payoff: ({payoff[0]}, {payoff[1]})"
        )

    # Compute aggregates
    total_agent_payoff = sum(r["payoff"][0] for r in rounds_data)
    coordination_count = sum(
        1 for r in rounds_data if r["canonical_action"] == r["opponent_action"]
    )

    result = {
        "game_id": game_id,
        "opponent_type": opponent.name,
        "canonical_history": history,
        "rounds": rounds_data,
        "total_agent_payoff": total_agent_payoff,
        "coordination_count": coordination_count,
    }

    print(
        f"    Game done. Agent total: {total_agent_payoff}, "
        f"Coordinated: {coordination_count}/{NUM_ROUNDS}"
    )

    return result
