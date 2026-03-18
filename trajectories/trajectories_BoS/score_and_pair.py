"""
score_and_pair.py — Rank scored trajectories and form DPO pairs.

Implements the top-25% / bottom-25% pairing strategy from §3 of the PDF.
Pairs are grouped by (opponent_type, prompt_type) so chosen/rejected
are always conditioned on the same game context.

Usage:
    python score_and_pair.py --input outputs/trajectories/trajectories_XYZ.jsonl
                             --config config.yaml
                             [--out-dir outputs/dpo_pairs]
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_trajectories(path: Path) -> list[dict]:
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    log.info(f"Loaded {len(trajectories)} trajectories from {path}")
    return trajectories


# ---------------------------------------------------------------------------
# Prompt reconstruction
# ---------------------------------------------------------------------------

def build_dpo_prompt(traj: dict, game_cfg: dict) -> str:
    a, b, miss = game_cfg["a"], game_cfg["b"], game_cfg["mismatch"]
    return (
        f"You are Player 1 in a {game_cfg['rounds']}-round Battle of the Sexes game.\n\n"
        f"Payoff matrix:\n"
        f"  Both Opera    -> ({a}, {b})  [you, opponent]\n"
        f"  Both Football -> ({b}, {a})  [you, opponent]\n"
        f"  Mismatch      -> ({miss}, {miss})\n\n"
        f"You prefer Opera. Your opponent is of type: {traj['opponent_type']}.\n"
        f"Play {game_cfg['rounds']} rounds to maximize your total payoff."
    )


def build_trajectory_text(traj: dict) -> str:
    """Render a trajectory as the 'chosen' or 'rejected' completion."""
    lines = []
    for r in traj["rounds"]:
        lines.append(f"Round {r['round_num']}:")
        lines.append(r["p1_reasoning"].strip())
        lines.append(
            f"Action: {r['p1_action']} | Opponent: {r['p2_action']} | Payoff: {r['p1_payoff']}"
        )
        lines.append("")
    lines.append(f"Total payoff: {traj['p1_total_payoff']:.1f}")
    lines.append(f"Coordination rate: {traj['coordination_rate']:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pairing logic
# ---------------------------------------------------------------------------

def form_pairs(
    trajectories: list[dict],
    top_pct: float = 0.25,
    bottom_pct: float = 0.25,
    game_cfg: dict = None,
    seed: int = 42,
) -> list[dict]:
    """
    Group trajectories by opponent_type, then within each group take the
    top X% as 'chosen' and bottom X% as 'rejected' to form DPO pairs.

    Returns a list of DPO pair dicts ready for training.
    """
    rng = random.Random(seed)

    # Group by opponent type
    groups: dict[str, list[dict]] = defaultdict(list)
    for traj in trajectories:
        groups[traj["opponent_type"]].append(traj)

    pairs = []
    for opp_type, group in groups.items():
        group_sorted = sorted(group, key=lambda t: t["score"], reverse=True)
        n = len(group_sorted)

        n_top = max(1, int(n * top_pct))
        n_bot = max(1, int(n * bottom_pct))

        chosen_pool = group_sorted[:n_top]
        rejected_pool = group_sorted[n - n_bot:]

        # Pair them up — shuffle both pools and zip
        rng.shuffle(chosen_pool)
        rng.shuffle(rejected_pool)

        n_pairs = min(len(chosen_pool), len(rejected_pool))
        log.info(f"  Opponent '{opp_type}': {n} trajectories -> {n_pairs} pairs "
                 f"(top score: {chosen_pool[0]['score']:.3f}, "
                 f"bot score: {rejected_pool[-1]['score']:.3f})")

        for chosen, rejected in zip(chosen_pool[:n_pairs], rejected_pool[:n_pairs]):
            prompt = build_dpo_prompt(chosen, game_cfg)
            pair = {
                "prompt": prompt,
                "chosen": build_trajectory_text(chosen),
                "rejected": build_trajectory_text(rejected),
                "metadata": {
                    "opponent_type": opp_type,
                    "chosen_score": chosen["score"],
                    "chosen_score_breakdown": chosen["score_breakdown"],
                    "chosen_prompt_type": chosen["p1_system_prompt_type"],
                    "chosen_coord_rate": chosen["coordination_rate"],
                    "chosen_p1_payoff": chosen["p1_total_payoff"],
                    "rejected_score": rejected["score"],
                    "rejected_score_breakdown": rejected["score_breakdown"],
                    "rejected_prompt_type": rejected["p1_system_prompt_type"],
                    "rejected_coord_rate": rejected["coordination_rate"],
                    "rejected_p1_payoff": rejected["p1_total_payoff"],
                    "score_gap": round(chosen["score"] - rejected["score"], 4),
                },
            }
            pairs.append(pair)

    log.info(f"Formed {len(pairs)} DPO pairs total.")
    return pairs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_pairs(pairs: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"dpo_pairs_{ts}.jsonl"
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    log.info(f"Saved {len(pairs)} DPO pairs to {out_path}")
    return out_path


def print_summary(pairs: list[dict]):
    if not pairs:
        log.warning("No pairs to summarize.")
        return
    gaps = [p["metadata"]["score_gap"] for p in pairs]
    avg_gap = sum(gaps) / len(gaps)
    by_opp: dict[str, int] = defaultdict(int)
    for p in pairs:
        by_opp[p["metadata"]["opponent_type"]] += 1

    log.info("\n=== DPO Pair Summary ===")
    log.info(f"Total pairs:          {len(pairs)}")
    log.info(f"Avg score gap:        {avg_gap:.4f}")
    log.info(f"Min score gap:        {min(gaps):.4f}")
    log.info(f"Max score gap:        {max(gaps):.4f}")
    log.info("Pairs per opponent type:")
    for opp, count in sorted(by_opp.items()):
        log.info(f"  {opp}: {count}")


# ---------------------------------------------------------------------------
# Sample pair printer (for sanity check)
# ---------------------------------------------------------------------------

def print_sample_pair(pair: dict):
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"OPPONENT: {pair['metadata']['opponent_type']}")
    print(f"Score gap: {pair['metadata']['score_gap']:.4f}  "
          f"(chosen: {pair['metadata']['chosen_score']:.3f}, "
          f"rejected: {pair['metadata']['rejected_score']:.3f})")
    print(f"\nPROMPT:\n{pair['prompt']}")
    print(f"\nCHOSEN (score={pair['metadata']['chosen_score']:.3f}):\n{pair['chosen'][:500]}...")
    print(f"\nREJECTED (score={pair['metadata']['rejected_score']:.3f}):\n{pair['rejected'][:500]}...")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Form DPO pairs from scored trajectories")
    parser.add_argument("--input", required=True, help="Path to trajectories JSONL file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-sample", action="store_true",
                        help="Print one sample pair to stdout for sanity check")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trajectories = load_trajectories(Path(args.input))

    out_dir = Path(args.out_dir or cfg["output"]["pairs_dir"])

    pairs = form_pairs(
        trajectories=trajectories,
        top_pct=cfg["pairing"]["top_pct"],
        bottom_pct=cfg["pairing"]["bottom_pct"],
        game_cfg=cfg["game"],
        seed=args.seed,
    )

    print_summary(pairs)

    if args.show_sample and pairs:
        import random
        print_sample_pair(random.choice(pairs))

    save_pairs(pairs, out_dir)


if __name__ == "__main__":
    main()
