"""
Main runner for Battle of the Sexes DPO trajectory generation.
Orchestrates: game play → scoring → labeling → export.

Usage:
    python main.py --mode test      # 2 trajectories per opponent
    python main.py --mode prod      # 50 trajectories per opponent
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
from agent import Agent
from opponents import get_opponent
from game_engine import run_game
from scoring import compute_final_score
from labeling import label_trajectories, export_dpo_dataset


def generate_all_trajectories(num_per_opponent: int) -> list[dict]:
    """Generate trajectories for all opponent types."""
    agent = Agent()
    all_trajectories = []

    total_games = num_per_opponent * len(OPPONENT_TYPES)
    game_counter = 0

    for opp_type in OPPONENT_TYPES:
        print(f"\n{'='*60}")
        print(f"Opponent: {opp_type}")
        print(f"{'='*60}")

        for i in range(num_per_opponent):
            game_counter += 1
            game_id = f"{opp_type}_{i}"
            print(f"\n  Game {game_counter}/{total_games}: {game_id}")

            opponent = get_opponent(opp_type)
            trajectory = run_game(agent, opponent, game_id=game_id)
            all_trajectories.append(trajectory)

    return all_trajectories


def score_all_trajectories(trajectories: list[dict]) -> list[dict]:
    """Score all trajectories and return scored trajectory list."""
    scored = []
    for i, traj in enumerate(trajectories):
        print(f"\n  Scoring trajectory {i + 1}/{len(trajectories)}: {traj['game_id']}...")
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


def main():
    parser = argparse.ArgumentParser(description="BoS DPO Trajectory Generator")
    parser.add_argument(
        "--mode",
        choices=["test", "prod"],
        default="test",
        help="test = 2 trajectories/opponent, prod = 50 trajectories/opponent",
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

    num_per_opponent = (
        TRAJECTORIES_PER_OPPONENT_TEST if args.mode == "test"
        else TRAJECTORIES_PER_OPPONENT_PROD
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Trajectories per opponent: {num_per_opponent}")
    print(f"Total games: {num_per_opponent * len(OPPONENT_TYPES)}")
    print(f"Output directory: {output_dir}")

    # ── Phase 1a: Generate trajectories ────────────────────────────
    if args.resume:
        print("\n" + "=" * 60)
        print(f"PHASE 1a: RESUMING from {args.resume}")
        print("=" * 60)
        with open(args.resume, "r") as f:
            trajectories = json.load(f)
        gen_time = 0.0
        print(f"Loaded {len(trajectories)} trajectories from file")
    else:
        print("\n" + "=" * 60)
        print("PHASE 1a: Generating trajectories")
        print("=" * 60)
        start_time = time.time()
        trajectories = generate_all_trajectories(num_per_opponent)
        gen_time = time.time() - start_time
        print(f"\nGenerated {len(trajectories)} trajectories in {gen_time:.1f}s")

        # Save raw trajectories
        raw_path = os.path.join(output_dir, "raw_trajectories.json")
        with open(raw_path, "w") as f:
            json.dump(trajectories, f, indent=2)
        print(f"Saved raw trajectories to {raw_path}")

    # ── Phase 1b: Score trajectories ───────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1b: Scoring trajectories")
    print("=" * 60)
    start_time = time.time()
    scored_trajectories = score_all_trajectories(trajectories)
    score_time = time.time() - start_time
    print(f"\nScored {len(scored_trajectories)} trajectories in {score_time:.1f}s")

    # Save scored trajectories
    scored_path = os.path.join(output_dir, "scored_trajectories.json")
    with open(scored_path, "w") as f:
        json.dump(scored_trajectories, f, indent=2)
    print(f"Saved scored trajectories to {scored_path}")

    # ── Phase 1c: Label trajectories ──────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1c: Labeling trajectories")
    print("=" * 60)
    labeled_data = label_trajectories(scored_trajectories)
    stats = labeled_data["stats"]
    print(f"  Total: {stats['total_trajectories']}")
    print(f"  Preferred (top 25%): {stats['num_preferred']}")
    print(f"  Non-preferred (bottom 25%): {stats['num_non_preferred']}")
    print(f"  Unlabeled (middle): {stats['num_unlabeled']}")
    print(f"  Preferred score range: {stats['preferred_score_range']}")
    print(f"  Non-preferred score range: {stats['non_preferred_score_range']}")

    # Save labeled data
    labeled_path = os.path.join(output_dir, "labeled_trajectories.json")
    with open(labeled_path, "w") as f:
        json.dump(labeled_data, f, indent=2)
    print(f"Saved labeled data to {labeled_path}")

    # ── Phase 1d: Export DPO pairs ────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1d: Exporting DPO pairs")
    print("=" * 60)
    dpo_path = os.path.join(output_dir, "dpo_pairs.jsonl")
    dpo_pairs = export_dpo_dataset(labeled_data, dpo_path)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  DPO pairs exported: {len(dpo_pairs)}")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Scoring time: {score_time:.1f}s")
    print(f"  Output directory: {output_dir}")

    # Print score distribution by opponent type
    print("\n  Score distribution by opponent type:")
    by_opp = {}
    for st in scored_trajectories:
        opp = st["trajectory"]["opponent_type"]
        if opp not in by_opp:
            by_opp[opp] = []
        by_opp[opp].append(st["scores"]["final_score"])

    for opp, scores in sorted(by_opp.items()):
        avg = sum(scores) / len(scores)
        print(f"    {opp}: avg={avg:.1f}, min={min(scores):.1f}, max={max(scores):.1f}")


if __name__ == "__main__":
    main()
