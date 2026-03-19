"""
Labeling module for DPO trajectory pairs.
Labels top 25% as preferred, bottom 25% as non-preferred.
Outputs in DPO-ready format.
"""

import json
from config import LABEL_TOP_PERCENTILE, LABEL_BOTTOM_PERCENTILE, NUM_ROUNDS, PAYOFFS


def label_trajectories(scored_trajectories: list[dict]) -> dict:
    """
    Given a list of scored trajectories (each with 'trajectory' and 'scores'),
    label the top and bottom percentiles.

    Returns a dict with:
      - "preferred": list of trajectories in top percentile
      - "non_preferred": list of trajectories in bottom percentile
      - "unlabeled": everything in between
      - "stats": summary statistics
    """
    # Sort by final score descending
    sorted_trajs = sorted(
        scored_trajectories,
        key=lambda x: x["scores"]["final_score"],
        reverse=True,
    )

    n = len(sorted_trajs)
    top_k = max(1, int(n * LABEL_TOP_PERCENTILE / 100))
    bottom_k = max(1, int(n * LABEL_BOTTOM_PERCENTILE / 100))

    preferred = sorted_trajs[:top_k]
    non_preferred = sorted_trajs[-bottom_k:]
    unlabeled = sorted_trajs[top_k: n - bottom_k] if n > top_k + bottom_k else []

    # Mark labels
    for t in preferred:
        t["label"] = "preferred"
    for t in non_preferred:
        t["label"] = "non_preferred"
    for t in unlabeled:
        t["label"] = "unlabeled"

    stats = {
        "total_trajectories": n,
        "num_preferred": len(preferred),
        "num_non_preferred": len(non_preferred),
        "num_unlabeled": len(unlabeled),
        "preferred_score_range": (
            preferred[-1]["scores"]["final_score"] if preferred else None,
            preferred[0]["scores"]["final_score"] if preferred else None,
        ),
        "non_preferred_score_range": (
            non_preferred[-1]["scores"]["final_score"] if non_preferred else None,
            non_preferred[0]["scores"]["final_score"] if non_preferred else None,
        ),
    }

    return {
        "preferred": preferred,
        "non_preferred": non_preferred,
        "unlabeled": unlabeled,
        "stats": stats,
    }


def format_trajectory_as_dpo_prompt(trajectory: dict) -> str:
    """
    Format a trajectory's agent reasoning/actions as the 'chosen' or 'rejected'
    field in a DPO training example.
    """
    lines = []
    for r in trajectory["rounds"]:
        lines.append(f"Round {r['round_num']}:")
        lines.append(f"Reasoning: {r['agent_reasoning']}")
        lines.append(f"Action: {r['agent_action']}")
        # Include what happened (outcome feedback)
        lines.append(
            f"Outcome: Opponent played {r['opponent_action']}. "
            f"Payoff: ({r['agent_payoff']}, {r['opponent_payoff']})"
        )
        lines.append("")
    return "\n".join(lines).strip()


def build_dpo_base_prompt() -> str:
    """Build the shared prompt for DPO pairs (game description only, option b)."""
    return (
        f"You are playing a {NUM_ROUNDS}-round repeated Battle of the Sexes game as Player 1. "
        f"You prefer Opera.\n\n"
        f"Payoff matrix:\n"
        f"  Both choose Opera:    you get 3, opponent gets 2\n"
        f"  You: Opera, Opponent: Football:  both get 0\n"
        f"  You: Football, Opponent: Opera:  both get 0\n"
        f"  Both choose Football: you get 2, opponent gets 3\n\n"
        f"You will play {NUM_ROUNDS} rounds. Each round, you see the history of previous "
        f"rounds and must choose Opera or Football. Your goal is to maximize your total "
        f"payoff across all rounds.\n\n"
        f"Before each action, reason about what your opponent is likely to do and why."
    )


def export_dpo_dataset(labeled_data: dict, output_path: str):
    """
    Export labeled trajectories as a JSONL file in DPO format.
    Each line is a JSON object with: prompt, chosen, rejected, metadata.
    """
    base_prompt = build_dpo_base_prompt()
    preferred = labeled_data["preferred"]
    non_preferred = labeled_data["non_preferred"]

    dpo_pairs = []

    # Pair preferred with non-preferred trajectories
    # Strategy: pair within the same opponent type when possible
    preferred_by_opp = _group_by_opponent(preferred)
    non_preferred_by_opp = _group_by_opponent(non_preferred)

    # First, pair within same opponent type
    for opp_type in preferred_by_opp:
        pref_list = preferred_by_opp[opp_type]
        non_pref_list = non_preferred_by_opp.get(opp_type, [])

        for i, pref in enumerate(pref_list):
            if i < len(non_pref_list):
                non_pref = non_pref_list[i]
            elif non_pref_list:
                # Reuse non-preferred if we run out
                non_pref = non_pref_list[i % len(non_pref_list)]
            else:
                # No non-preferred for this opponent type; skip or use cross-type
                continue

            pair = {
                "prompt": base_prompt,
                "chosen": format_trajectory_as_dpo_prompt(pref["trajectory"]),
                "rejected": format_trajectory_as_dpo_prompt(non_pref["trajectory"]),
                "metadata": {
                    "chosen_opponent": pref["trajectory"]["opponent_type"],
                    "rejected_opponent": non_pref["trajectory"]["opponent_type"],
                    "chosen_score": pref["scores"]["final_score"],
                    "rejected_score": non_pref["scores"]["final_score"],
                    "chosen_game_id": pref["trajectory"]["game_id"],
                    "rejected_game_id": non_pref["trajectory"]["game_id"],
                },
            }
            dpo_pairs.append(pair)

    # Write JSONL
    with open(output_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Exported {len(dpo_pairs)} DPO pairs to {output_path}")
    return dpo_pairs


def _group_by_opponent(trajectories: list[dict]) -> dict:
    """Group trajectories by opponent type."""
    groups = {}
    for t in trajectories:
        opp = t["trajectory"]["opponent_type"]
        if opp not in groups:
            groups[opp] = []
        groups[opp].append(t)
    return groups
