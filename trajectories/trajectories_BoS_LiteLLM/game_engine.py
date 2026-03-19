"""
Game engine for running a single Battle of the Sexes game.
Collects full trajectory data including actions, payoffs, and reasoning.
"""

from config import PAYOFFS, NUM_ROUNDS


def run_game(agent, opponent, game_id: str = "0") -> dict:
    """
    Run a single repeated Battle of the Sexes game.

    Returns a trajectory dict:
    {
        "game_id": str,
        "opponent_type": str,
        "rounds": [
            {
                "round_num": int,
                "agent_action": str,
                "opponent_action": str,
                "agent_payoff": int,
                "opponent_payoff": int,
                "agent_reasoning": str,
                "agent_raw_response": str,
            },
            ...
        ],
        "total_agent_payoff": int,
        "total_opponent_payoff": int,
        "coordination_count": int,
    }
    """
    history = []  # list of {"agent": action, "opponent": action}
    rounds_data = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"    Round {round_num}/{NUM_ROUNDS}...")

        # Agent chooses action (with reasoning)
        agent_result = agent.choose_action(history, round_num)
        agent_action = agent_result["action"]

        # Opponent chooses action
        opponent_action = opponent.choose_action(history)

        # Compute payoffs
        payoff = PAYOFFS[(agent_action, opponent_action)]
        agent_payoff, opponent_payoff = payoff

        # Record round data
        round_data = {
            "round_num": round_num,
            "agent_action": agent_action,
            "opponent_action": opponent_action,
            "agent_payoff": agent_payoff,
            "opponent_payoff": opponent_payoff,
            "agent_reasoning": agent_result["reasoning"],
            "agent_raw_response": agent_result["raw_response"],
        }
        rounds_data.append(round_data)

        # Update history (visible to both players next round)
        history.append({
            "agent": agent_action,
            "opponent": opponent_action,
        })

        print(
            f"      Agent: {agent_action}, Opponent: {opponent_action} "
            f"→ Payoffs: ({agent_payoff}, {opponent_payoff})"
        )

    # Compute aggregates
    total_agent_payoff = sum(r["agent_payoff"] for r in rounds_data)
    total_opponent_payoff = sum(r["opponent_payoff"] for r in rounds_data)
    coordination_count = sum(
        1 for r in rounds_data if r["agent_action"] == r["opponent_action"]
    )

    trajectory = {
        "game_id": game_id,
        "opponent_type": opponent.name,
        "rounds": rounds_data,
        "total_agent_payoff": total_agent_payoff,
        "total_opponent_payoff": total_opponent_payoff,
        "coordination_count": coordination_count,
    }

    print(
        f"    Game done. Agent total: {total_agent_payoff}, "
        f"Coordinated: {coordination_count}/{NUM_ROUNDS}"
    )

    return trajectory
