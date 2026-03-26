"""
Main runner for Battle of the Sexes DPO trajectory generation.

Pipeline per opponent type:
  1. Level_2 agent plays N live games against the opponent
  2. At each round, level_0 responds to the same history (hypothetical)
  3. DPO pairs formed: chosen = level_2, rejected = level_0
  4. (Optional) Validate a sample with LLM judge

Usage:
    python main.py --mode test                    # 1 game per opponent
    python main.py --mode prod                    # 15 games per opponent
    python main.py --mode test --validate         # with validation scoring
    python main.py --mode test --validate-only output/run_XXX/raw_games.json
"""

import argparse
import json
import os
import time
from datetime import datetime

from config import (
    OPPONENT_TYPES, GAMES_PER_OPPONENT_TEST, GAMES_PER_OPPONENT_PROD, NUM_ROUNDS,
)
from opponents import get_opponent
from game_engine import run_game_and_collect_dpo
from scoring import validate_game


def main():
    parser = argparse.ArgumentParser(description="BoS DPO Data Generator")
    parser.add_argument(
        "--mode", choices=["test", "prod"], default="test",
        help="test = 1 game/opponent, prod = 15 games/opponent",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to save results",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation scoring on generated data",
    )
    parser.add_argument(
        "--validate-only", default=None,
        help="Path to existing raw_games.json — skip generation, only run validation",
    )
    args = parser.parse_args()

    def get_num_games(opp_type):
        if args.mode == "test":
            return GAMES_PER_OPPONENT_TEST
        return GAMES_PER_OPPONENT_PROD.get(opp_type, 3)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    total_games = sum(get_num_games(opp) for opp in OPPONENT_TYPES)
    total_dpo_pairs = total_games * NUM_ROUNDS

    print(f"Mode: {args.mode}")
    if args.mode == "prod":
        print(f"Games per opponent: {GAMES_PER_OPPONENT_PROD}")
    else:
        print(f"Games per opponent: {GAMES_PER_OPPONENT_TEST} (test mode)")
    print(f"Opponent types: {OPPONENT_TYPES}")
    print(f"Total games: {total_games}")
    print(f"Expected DPO pairs: {total_dpo_pairs}")
    print(f"Output directory: {output_dir}")
    print(f"Validation: {'yes' if args.validate else 'no'}")

    # ── Load or generate ───────────────────────────────────────────
    if args.validate_only:
        print(f"\nLoading from {args.validate_only}...")
        with open(args.validate_only, "r") as f:
            all_games = json.load(f)
        print(f"Loaded {len(all_games)} games")
        gen_time = 0.0
    else:
        all_games = []
        game_counter = 0
        gen_start = time.time()

        for opp_type in OPPONENT_TYPES:
            num_games = get_num_games(opp_type)
            print(f"\n{'='*60}")
            print(f"OPPONENT: {opp_type} ({num_games} games)")
            print(f"{'='*60}")

            for i in range(num_games):
                game_counter += 1
                game_id = f"{opp_type}_{i}"
                print(f"\n  Game {game_counter}/{total_games}: {game_id}")

                opponent = get_opponent(opp_type)
                game_data = run_game_and_collect_dpo(opponent, game_id=game_id)
                all_games.append(game_data)

        gen_time = time.time() - gen_start
        print(f"\nGenerated {len(all_games)} games in {gen_time:.1f}s")

        # Save raw games
        raw_path = os.path.join(output_dir, "raw_games.json")
        with open(raw_path, "w") as f:
            json.dump(all_games, f, indent=2)
        print(f"Saved raw games to {raw_path}")

    # ── Extract DPO pairs ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXTRACTING DPO PAIRS")
    print(f"{'='*60}")

    dpo_pairs = []
    for game in all_games:
        for round_data in game["rounds"]:
            pair = {
                "prompt": round_data["dpo_prompt"],
                "chosen": (
                    f"{round_data['chosen']['reasoning']}\n"
                    f"Action: {round_data['chosen']['action']}"
                ),
                "rejected": (
                    f"{round_data['rejected']['reasoning']}\n"
                    f"Action: {round_data['rejected']['action']}"
                ),
                "metadata": {
                    "game_id": game["game_id"],
                    "opponent_type": game["opponent_type"],
                    "round_num": round_data["round_num"],
                    "chosen_action": round_data["chosen"]["action"],
                    "rejected_action": round_data["rejected"]["action"],
                    "canonical_action": round_data["canonical_action"],
                    "opponent_action": round_data["opponent_action"],
                },
            }
            dpo_pairs.append(pair)

    # Save DPO pairs
    dpo_path = os.path.join(output_dir, "dpo_pairs.jsonl")
    with open(dpo_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Exported {len(dpo_pairs)} DPO pairs to {dpo_path}")

    # ── Optional validation ────────────────────────────────────────
    if args.validate or args.validate_only:
        print(f"\n{'='*60}")
        print("VALIDATION SCORING")
        print(f"{'='*60}")

        # In test mode validate all, in prod validate a sample
        games_to_validate = all_games
        if args.mode == "prod" and not args.validate_only:
            # Sample 1 game per opponent type for validation
            seen_opps = set()
            games_to_validate = []
            for g in all_games:
                if g["opponent_type"] not in seen_opps:
                    games_to_validate.append(g)
                    seen_opps.add(g["opponent_type"])

        print(f"Validating {len(games_to_validate)} games...")

        validation_results = []
        for game in games_to_validate:
            print(f"\n    Validating game: {game['game_id']}")
            result = validate_game(game)
            validation_results.append(result)
            print(
                f"      Chosen avg: {result['chosen_avg_score']:.1f}, "
                f"Rejected avg: {result['rejected_avg_score']:.1f}, "
                f"Chosen win rate: {result['chosen_win_rate']:.0%}"
            )

        # Save validation results
        val_path = os.path.join(output_dir, "validation_results.json")
        with open(val_path, "w") as f:
            json.dump(validation_results, f, indent=2)
        print(f"\nSaved validation results to {val_path}")

        # Validation summary
        all_chosen_avgs = [r["chosen_avg_score"] for r in validation_results]
        all_rejected_avgs = [r["rejected_avg_score"] for r in validation_results]
        all_win_rates = [r["chosen_win_rate"] for r in validation_results]

        print(f"\n  Validation Summary:")
        print(f"    Chosen avg score:  {sum(all_chosen_avgs)/len(all_chosen_avgs):.1f}")
        print(f"    Rejected avg score: {sum(all_rejected_avgs)/len(all_rejected_avgs):.1f}")
        print(f"    Chosen win rate:   {sum(all_win_rates)/len(all_win_rates):.0%}")

        print(f"\n  Per opponent type:")
        for r in validation_results:
            print(
                f"    {r['opponent_type']}: "
                f"chosen={r['chosen_avg_score']:.1f}, "
                f"rejected={r['rejected_avg_score']:.1f}, "
                f"win_rate={r['chosen_win_rate']:.0%}"
            )

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total games: {len(all_games)}")
    print(f"  Total DPO pairs: {len(dpo_pairs)}")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Output directory: {output_dir}")

    # Per opponent breakdown
    print(f"\n  Per opponent type:")
    by_opp = {}
    for game in all_games:
        opp = game["opponent_type"]
        if opp not in by_opp:
            by_opp[opp] = {"games": 0, "pairs": 0, "total_payoff": 0, "total_coord": 0}
        by_opp[opp]["games"] += 1
        by_opp[opp]["pairs"] += len(game["rounds"])
        by_opp[opp]["total_payoff"] += game["total_agent_payoff"]
        by_opp[opp]["total_coord"] += game["coordination_count"]

    for opp, stats in sorted(by_opp.items()):
        avg_payoff = stats["total_payoff"] / stats["games"]
        avg_coord = stats["total_coord"] / stats["games"]
        print(
            f"    {opp}: {stats['games']} games, {stats['pairs']} pairs, "
            f"avg_payoff={avg_payoff:.1f}, avg_coord={avg_coord:.1f}/{NUM_ROUNDS}"
        )


if __name__ == "__main__":
    main()
