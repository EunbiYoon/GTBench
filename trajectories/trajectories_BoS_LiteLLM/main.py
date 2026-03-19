"""
Main runner for Battle of the Sexes DPO trajectory generation.
Processes per opponent type: generate → score → label → pair, then next opponent.

Usage:
    python main.py --mode test      # 1 trajectory per opponent per level
    python main.py --mode prod      # 50 trajectories per opponent per level
    python main.py --mode test --resume output/run_XXXX/raw_trajectories.json
"""

import argparse
import json
import os
import time
from datetime import datetime

from config import (
    OPPONENT_TYPES,
    TRAJECTORIES_PER_OPPONENT_TEST,
    TRAJECTORIES_PER_OPPONENT_PROD,
)
from agent import Agent, VALID_LEVELS
from opponents import get_opponent
from game_engine import run_game
from scoring import compute_final_score
from labeling import (
    label_trajectories,
    build_dpo_base_prompt,
    format_trajectory_as_dpo_prompt,
)


def generate_trajectories_for_opponent(opp_type: str, num_per_level: int, game_counter: int, total_games: int) -> tuple[list[dict], int]:
    """Generate trajectories for one opponent type across all reasoning levels."""
    trajectories = []

    for level in VALID_LEVELS:
        print(f"\n  --- Reasoning Level: {level} ---")
        agent = Agent(reasoning_level=level)

        for i in range(num_per_level):
            game_counter += 1
            game_id = f"{opp_type}_{level}_{i}"
            print(f"\n  Game {game_counter}/{total_games}: {game_id}")

            opponent = get_opponent(opp_type)
            trajectory = run_game(agent, opponent, game_id=game_id)
            trajectory["reasoning_level"] = level
            trajectories.append(trajectory)

    return trajectories, game_counter


def score_trajectories(trajectories: list[dict], label: str = "") -> list[dict]:
    """Score a list of trajectories."""
    scored = []
    for i, traj in enumerate(trajectories):
        prefix = f"[{label}] " if label else ""
        print(f"\n  {prefix}Scoring {i + 1}/{len(trajectories)}: {traj['game_id']}...")
        scores = compute_final_score(traj)
        scored.append({
            "trajectory": traj,
            "scores": scores,
        })
        print(
            f"    Coordination: {scores['coordination_score']}, "
            f"Payoff: {scores['payoff_score']}, "
            f"Reasoning: {scores['reasoning_score']}, "
            f"FINAL: {scores['final_score']}"
        )
    return scored


def form_dpo_pairs_for_opponent(labeled_data: dict, opp_type: str) -> list[dict]:
    """
    Form DPO pairs from labeled trajectories for a single opponent type.
    Pairs each preferred trajectory with each non-preferred trajectory.
    """
    base_prompt = build_dpo_base_prompt()
    preferred = labeled_data["preferred"]
    non_preferred = labeled_data["non_preferred"]
    pairs = []

    for pref in preferred:
        for non_pref in non_preferred:
            pair = {
                "prompt": base_prompt,
                "chosen": format_trajectory_as_dpo_prompt(pref["trajectory"]),
                "rejected": format_trajectory_as_dpo_prompt(non_pref["trajectory"]),
                "metadata": {
                    "opponent_type": opp_type,
                    "chosen_reasoning_level": pref["trajectory"].get("reasoning_level", "unknown"),
                    "rejected_reasoning_level": non_pref["trajectory"].get("reasoning_level", "unknown"),
                    "chosen_score": pref["scores"]["final_score"],
                    "rejected_score": non_pref["scores"]["final_score"],
                    "chosen_game_id": pref["trajectory"]["game_id"],
                    "rejected_game_id": non_pref["trajectory"]["game_id"],
                },
            }
            pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="BoS DPO Trajectory Generator")
    parser.add_argument(
        "--mode",
        choices=["test", "prod"],
        default="test",
        help="test = small run, prod = full run",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save results",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to existing raw_trajectories.json to resume from scoring phase",
    )
    args = parser.parse_args()

    num_per_level = (
        TRAJECTORIES_PER_OPPONENT_TEST if args.mode == "test"
        else TRAJECTORIES_PER_OPPONENT_PROD
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    total_games = num_per_level * len(OPPONENT_TYPES) * len(VALID_LEVELS)

    print(f"Mode: {args.mode}")
    print(f"Trajectories per opponent per level: {num_per_level}")
    print(f"Reasoning levels: {VALID_LEVELS}")
    print(f"Opponent types: {OPPONENT_TYPES}")
    print(f"Total games: {total_games}")
    print(f"Output directory: {output_dir}")

    # ── Handle resume ──────────────────────────────────────────────
    if args.resume:
        print(f"\nResuming from {args.resume} — skipping generation, running scoring + labeling per opponent.")
        with open(args.resume, "r") as f:
            all_trajectories = json.load(f)
        print(f"Loaded {len(all_trajectories)} trajectories")

        # Group by opponent type
        trajs_by_opp = {}
        for t in all_trajectories:
            opp = t["opponent_type"]
            if opp not in trajs_by_opp:
                trajs_by_opp[opp] = []
            trajs_by_opp[opp].append(t)

        gen_time = 0.0
    else:
        trajs_by_opp = None  # will be built during generation

    # ── Per-opponent pipeline ──────────────────────────────────────
    all_trajectories = []
    all_scored = []
    all_dpo_pairs = []
    all_labeled_data = {}
    game_counter = 0
    gen_time = 0.0
    score_time = 0.0

    for opp_type in OPPONENT_TYPES:
        print(f"\n{'='*60}")
        print(f"OPPONENT: {opp_type}")
        print(f"{'='*60}")

        # ── Generate (or load) trajectories for this opponent ──────
        if trajs_by_opp is not None:
            # Resume mode: use loaded trajectories
            opp_trajectories = trajs_by_opp.get(opp_type, [])
            print(f"  Loaded {len(opp_trajectories)} trajectories from file")
        else:
            # Generate fresh
            print(f"\n  Generating trajectories...")
            start = time.time()
            opp_trajectories, game_counter = generate_trajectories_for_opponent(
                opp_type, num_per_level, game_counter, total_games
            )
            elapsed = time.time() - start
            gen_time += elapsed
            print(f"  Generated {len(opp_trajectories)} trajectories in {elapsed:.1f}s")

        all_trajectories.extend(opp_trajectories)

        # ── Score trajectories for this opponent ───────────────────
        print(f"\n  Scoring trajectories for {opp_type}...")
        start = time.time()
        scored = score_trajectories(opp_trajectories, label=opp_type)
        elapsed = time.time() - start
        score_time += elapsed
        all_scored.extend(scored)

        # ── Label: top 25% preferred, bottom 25% non-preferred ─────
        print(f"\n  Labeling trajectories for {opp_type}...")
        labeled = label_trajectories(scored)
        stats = labeled["stats"]
        print(f"    Total: {stats['total_trajectories']}")
        print(f"    Preferred: {stats['num_preferred']} (scores: {stats['preferred_score_range']})")
        print(f"    Non-preferred: {stats['num_non_preferred']} (scores: {stats['non_preferred_score_range']})")
        print(f"    Unlabeled: {stats['num_unlabeled']}")
        all_labeled_data[opp_type] = labeled

        # ── Form DPO pairs for this opponent ───────────────────────
        pairs = form_dpo_pairs_for_opponent(labeled, opp_type)
        all_dpo_pairs.extend(pairs)
        print(f"    DPO pairs formed: {len(pairs)}")

    # ── Save all outputs ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")

    # Raw trajectories
    raw_path = os.path.join(output_dir, "raw_trajectories.json")
    with open(raw_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"  Raw trajectories: {raw_path}")

    # Scored trajectories
    scored_path = os.path.join(output_dir, "scored_trajectories.json")
    with open(scored_path, "w") as f:
        json.dump(all_scored, f, indent=2)
    print(f"  Scored trajectories: {scored_path}")

    # Labeled data (per opponent type)
    labeled_path = os.path.join(output_dir, "labeled_trajectories.json")
    with open(labeled_path, "w") as f:
        # Convert to serializable format
        serializable = {}
        for opp, data in all_labeled_data.items():
            serializable[opp] = data
        json.dump(serializable, f, indent=2)
    print(f"  Labeled trajectories: {labeled_path}")

    # DPO pairs (JSONL)
    dpo_path = os.path.join(output_dir, "dpo_pairs.jsonl")
    with open(dpo_path, "w") as f:
        for pair in all_dpo_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"  DPO pairs: {dpo_path} ({len(all_dpo_pairs)} pairs)")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total trajectories: {len(all_trajectories)}")
    print(f"  Total DPO pairs: {len(all_dpo_pairs)}")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Scoring time: {score_time:.1f}s")
    print(f"  Output directory: {output_dir}")

    # Score distribution by opponent type
    print("\n  Score distribution by opponent type:")
    by_opp = {}
    for st in all_scored:
        opp = st["trajectory"]["opponent_type"]
        if opp not in by_opp:
            by_opp[opp] = []
        by_opp[opp].append(st["scores"]["final_score"])
    for opp, scores in sorted(by_opp.items()):
        avg = sum(scores) / len(scores)
        print(f"    {opp}: avg={avg:.1f}, min={min(scores):.1f}, max={max(scores):.1f}")

    # Score distribution by reasoning level
    print("\n  Score distribution by reasoning level:")
    by_level = {}
    for st in all_scored:
        level = st["trajectory"].get("reasoning_level", "unknown")
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(st["scores"]["final_score"])
    for level, scores in sorted(by_level.items()):
        avg = sum(scores) / len(scores)
        print(f"    {level}: avg={avg:.1f}, min={min(scores):.1f}, max={max(scores):.1f}")

    # DPO pairs by opponent type
    print("\n  DPO pairs by opponent type:")
    pairs_by_opp = {}
    for p in all_dpo_pairs:
        opp = p["metadata"]["opponent_type"]
        pairs_by_opp[opp] = pairs_by_opp.get(opp, 0) + 1
    for opp, count in sorted(pairs_by_opp.items()):
        print(f"    {opp}: {count} pairs")


if __name__ == "__main__":
    main()
