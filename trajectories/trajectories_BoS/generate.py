"""
generate.py — Trajectory generation using vLLM (offline inference).

Runs P1 (LLM) against typed opponents across multiple episodes.
Saves raw trajectories to JSONL before scoring.

Usage:
    python generate.py --config config.yaml [--seed 42] [--debug]
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import time
from pathlib import Path

import yaml
from vllm import LLM, SamplingParams

from game import BoSConfig, RoundResult, Trajectory, parse_action, score_trajectory
from opponents import build_opponent_pool
from prompts import SYSTEM_PROMPTS, build_round_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def load_model(model_cfg: dict) -> tuple[LLM, SamplingParams]:
    log.info(f"Loading model: {model_cfg['name']}")
    llm = LLM(
        model=model_cfg["name"],
        tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        trust_remote_code=True,
        dtype=model_cfg.get("dtype", "auto"),
    )
    sampling = SamplingParams(
        temperature=model_cfg.get("temperature", 0.8),
        top_p=model_cfg.get("top_p", 0.95),
        max_tokens=model_cfg.get("max_tokens", 512),
    )
    log.info("Model loaded.")
    return llm, sampling


def build_chat_prompt(tokenizer, system: str, messages: list[dict]) -> str:
    """
    Convert system + user/assistant turns into a single string using the
    model's chat template. Falls back to manual formatting if no template.
    """
    full = [{"role": "system", "content": system}] + messages
    try:
        return tokenizer.apply_chat_template(
            full, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: simple format
        parts = [f"<|system|>\n{system}\n"]
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Single-episode generation
# ---------------------------------------------------------------------------

def run_episode(
    llm: LLM,
    sampling: SamplingParams,
    tokenizer,
    opponent,
    system_prompt: str,
    game_cfg: BoSConfig,
    reveal_opponent_type: bool = False,
) -> Trajectory:
    """
    Run one full episode (game_cfg.rounds rounds) of P1 vs opponent.
    Returns an unscored Trajectory.
    """
    traj = Trajectory(
        opponent_type=opponent.name,
        p1_system_prompt_type="",  # filled by caller
    )
    opponent.reset()

    p1_history: list = []
    opp_history: list = []
    p1_payoffs: list = []
    conversation: list[dict] = []

    for rnd in range(1, game_cfg.rounds + 1):
        user_msg = build_round_prompt(
            round_num=rnd,
            total_rounds=game_cfg.rounds,
            opponent_history=opp_history,
            p1_history=p1_history,
            p1_payoffs=p1_payoffs,
            opponent_type_hint=opponent.name if reveal_opponent_type else None,
        )
        conversation.append({"role": "user", "content": user_msg})

        prompt_str = build_chat_prompt(tokenizer, system_prompt, conversation)
        output = llm.generate([prompt_str], sampling)[0]
        assistant_text = output.outputs[0].text.strip()

        conversation.append({"role": "assistant", "content": assistant_text})

        # Parse P1 action
        p1_action = parse_action(assistant_text)
        if p1_action is None:
            log.warning(f"Round {rnd}: Could not parse P1 action from: {assistant_text[:80]!r}. Defaulting to Opera.")
            p1_action = "Opera"

        # Get opponent action (conditioned on P1's history so far)
        p2_action = opponent.act(p1_history)

        p1_pay, p2_pay = game_cfg.payoff(p1_action, p2_action)

        result = RoundResult(
            round_num=rnd,
            p1_action=p1_action,
            p2_action=p2_action,
            p1_payoff=p1_pay,
            p2_payoff=p2_pay,
            p1_reasoning=assistant_text,
            is_coordinated=(p1_action == p2_action),
            is_p1_preferred_ne=(p1_action == "Opera" and p2_action == "Opera"),
            is_p2_preferred_ne=(p1_action == "Football" and p2_action == "Football"),
        )
        traj.rounds.append(result)
        p1_history.append(p1_action)
        opp_history.append(p2_action)
        p1_payoffs.append(p1_pay)

    return traj


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_all(cfg: dict, seed: int = 42) -> list[Trajectory]:
    rng = random.Random(seed)

    game_cfg = BoSConfig(
        a=cfg["game"]["a"],
        b=cfg["game"]["b"],
        mismatch=cfg["game"]["mismatch"],
        rounds=cfg["game"]["rounds"],
    )

    llm, sampling = load_model(cfg["model"])

    # Access tokenizer through llm
    tokenizer = llm.get_tokenizer()

    opponent_pool = build_opponent_pool(
        cfg["opponents"]["types"],
        epsilon=cfg["opponents"]["epsilon"],
        seed=seed,
    )

    prompt_types = cfg["generation"]["p1_system_prompts"]
    n_per_opponent = cfg["generation"]["trajectories_per_opponent"]
    scoring_weights = cfg["scoring"]

    trajectories: list[Trajectory] = []
    total = len(opponent_pool) * n_per_opponent
    log.info(f"Generating {total} trajectories ({len(opponent_pool)} opponents × {n_per_opponent} episodes).")

    for opponent in opponent_pool:
        log.info(f"Opponent: {opponent.name} ({n_per_opponent} episodes)")

        for ep in range(n_per_opponent):
            # Sample a prompt type according to weights
            prompt_cfg = rng.choices(
                prompt_types,
                weights=[p["weight"] for p in prompt_types],
            )[0]
            prompt_type = prompt_cfg["type"]

            system_fn = SYSTEM_PROMPTS[prompt_type]
            system_prompt = system_fn(game_cfg)

            traj = run_episode(
                llm=llm,
                sampling=sampling,
                tokenizer=tokenizer,
                opponent=opponent,
                system_prompt=system_prompt,
                game_cfg=game_cfg,
            )
            traj.p1_system_prompt_type = prompt_type

            traj = score_trajectory(traj, game_cfg, scoring_weights)
            trajectories.append(traj)

            if (ep + 1) % 5 == 0:
                log.info(f"  Episode {ep+1}/{n_per_opponent} done. "
                         f"Last score: {traj.score:.3f}, "
                         f"payoff: {traj.p1_total_payoff:.1f}, "
                         f"coord_rate: {traj.coordination_rate:.2f}")

    return trajectories


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_trajectories(trajectories: list[Trajectory], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"trajectories_{ts}.jsonl"
    with open(out_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict()) + "\n")
    log.info(f"Saved {len(trajectories)} trajectories to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate BoS trajectories with vLLM")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true",
                        help="Generate only 2 trajectories per opponent for quick test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["generation"]["trajectories_per_opponent"] = 2
        log.info("Debug mode: 2 trajectories per opponent.")

    trajectories = generate_all(cfg, seed=args.seed)

    out_dir = Path(cfg["output"]["trajectories_dir"])
    save_trajectories(trajectories, out_dir)

    # Quick summary
    scores = [t.score for t in trajectories]
    log.info(f"\n=== Summary ===")
    log.info(f"Total trajectories: {len(trajectories)}")
    log.info(f"Score  mean: {sum(scores)/len(scores):.3f}")
    log.info(f"Score  max:  {max(scores):.3f}")
    log.info(f"Score  min:  {min(scores):.3f}")
    avg_coord = sum(t.coordination_rate for t in trajectories) / len(trajectories)
    log.info(f"Avg coordination rate: {avg_coord:.3f}")


if __name__ == "__main__":
    main()
