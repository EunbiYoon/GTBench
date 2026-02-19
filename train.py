"""
train.py — PPO fine-tuning of TinyLlama on Battle of the Sexes
==============================================================
What this does:
  - Loads TinyLlama-1.1B as the policy LLM
  - Each training step: plays one full BoS episode (10 rounds)
  - Each round: LLM reads the text prompt, outputs "Football" or "Ballet"
  - Reward signal: game payoff (+ coordination bonus)
  - PPO updates the LLM weights to maximize cumulative reward

How TRL's PPO works for LLMs:
  - Keeps a frozen REFERENCE model (the original TinyLlama)
  - Adds a KL penalty to stop the policy drifting too far from reference
  - This prevents the LLM from "forgetting" language while learning the game

Run on Unity cluster:
  sbatch job.sh   (see job.sh for SLURM config)
  or directly:
  python train.py
"""

import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model

from game   import BoSGame, Alternator, N_ROUNDS, TRAIN_MATRICES, TEST_MATRICES
from prompt import build_prompt, ACTION_TOKENS, ACTION_TO_ID


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REWARD_MODE = "coordinated"   # "selfish" or "coordinated"
N_EPISODES  = 2000            # total training episodes
EVAL_EVERY  = 200             # evaluate against Alternator every N episodes
SAVE_PATH   = "./tinyllama_bos_ppo"

PPO_CFG = PPOConfig(
    model_name              = MODEL_NAME,
    learning_rate           = 1e-5,
    batch_size              = 8,       # episodes per PPO update
    mini_batch_size         = 4,
    gradient_accumulation_steps = 1,
    kl_penalty              = "kl",    # keep LLM from drifting too far
    init_kl_coef            = 0.1,
    target_kl               = 6.0,
    ppo_epochs              = 4,
    log_with                = None,    # set to "wandb" if you want tracking
)


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing — extract Football/Ballet from LLM output
# ─────────────────────────────────────────────────────────────────────────────

def parse_action(text: str) -> int:
    """
    Parse the LLM's output text into action 0 (Football) or 1 (Ballet).
    Falls back to random if the LLM outputs something unexpected.

    Examples:
      " Football"  → 0
      " Ballet\n"  → 1
      "I choose Football because..." → 0  (substring match)
    """
    text = text.strip().lower()
    if "football" in text:
        return 0
    elif "ballet" in text:
        return 1
    else:
        return random.randint(0, 1)   # fallback


# ─────────────────────────────────────────────────────────────────────────────
# LLM action: feed prompt → get token → parse action
# ─────────────────────────────────────────────────────────────────────────────

def llm_act(model, tokenizer, prompt: str, device) -> tuple[int, torch.Tensor]:
    """
    Run one forward pass through the LLM.

    Returns:
        action      : int (0=Football, 1=Ballet)
        response_ids: token ids of the generated response (for PPO)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = 5,        # just need one word
            do_sample      = True,     # sample so PPO gets probability info
            temperature    = 0.7,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # only the newly generated tokens (not the prompt)
    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    action = parse_action(response_text)

    return action, response_ids


# ─────────────────────────────────────────────────────────────────────────────
# Play one full episode (10 rounds) and collect PPO data
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(model, tokenizer, device, opponent=None, matrix=None):
    """
    Play one full BoS episode.

    Returns:
        queries   : list of tokenized prompts (one per round)
        responses : list of tokenized LLM outputs
        rewards   : list of per-round reward tensors
        coord_rate: fraction of rounds where actions matched
    """
    game    = BoSGame(opponent=opponent, matrix=matrix, reward_mode=REWARD_MODE)
    queries, responses, rewards = [], [], []
    coordinated_rounds = 0

    while not game.is_done:
        # 1. Build text prompt from current game state
        prompt = build_prompt(game.AB, game.history, N_ROUNDS)

        # 2. LLM reads prompt, outputs action
        action, response_ids = llm_act(model, tokenizer, prompt, device)

        # 3. Step the game, get reward
        reward, info = game.step(action)

        if info["coordinated"]:
            coordinated_rounds += 1

        # 4. Tokenize the prompt for PPO
        query_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)

        queries.append(query_ids)
        responses.append(response_ids.to(device))
        rewards.append(torch.tensor(reward, dtype=torch.float))

    coord_rate = coordinated_rounds / N_ROUNDS
    return queries, responses, rewards, coord_rate


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: run against Alternator, report coordination rate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, device, n_episodes=20):
    """
    Evaluate against the Alternator opponent (the key test from the paper).
    Paper baseline: GPT-4 gets ~50% (basically fails to coordinate).
    """
    coord_rates = []

    for _ in range(n_episodes):
        # use canonical matrix for fair comparison with paper
        _, _, _, coord_rate = run_episode(
            model, tokenizer, device,
            opponent=Alternator(),
            matrix=(10, 7),
        )
        coord_rates.append(coord_rate)

    avg = sum(coord_rates) / len(coord_rates)
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model and tokenizer ──────────────────────────────────────────────
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # TRL wraps the model with a value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
    model = model.to(device)

    # Frozen reference model — PPO uses this for KL penalty
    ref_model = create_reference_model(model)

    ppo_trainer = PPOTrainer(
        config    = PPO_CFG,
        model     = model,
        ref_model = ref_model,
        tokenizer = tokenizer,
    )

    # ── Evaluate before any training (baseline) ───────────────────────────────
    print("\n── Baseline (before training) ──")
    baseline_coord = evaluate(model, tokenizer, device)
    print(f"  Coordination rate vs Alternator: {baseline_coord:.1%}")
    print(f"  Paper's GPT-4 baseline:          ~50%")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n── Training for {N_EPISODES} episodes ──")

    all_queries, all_responses, all_rewards = [], [], []
    episode_coord_rates = []

    for episode in range(N_EPISODES):

        queries, responses, rewards, coord_rate = run_episode(
            model, tokenizer, device
        )
        episode_coord_rates.append(coord_rate)

        all_queries.extend(queries)
        all_responses.extend(responses)
        all_rewards.extend(rewards)

        # ── PPO update every batch_size episodes ─────────────────────────────
        if len(all_queries) >= PPO_CFG.batch_size * N_ROUNDS:
            stats = ppo_trainer.step(all_queries, all_responses, all_rewards)
            all_queries, all_responses, all_rewards = [], [], []

            avg_coord = sum(episode_coord_rates) / len(episode_coord_rates)
            episode_coord_rates = []
            print(f"  Episode {episode+1:>4} │ "
                  f"avg coord rate (last batch): {avg_coord:.1%} │ "
                  f"mean reward: {stats['ppo/mean_scores']:.2f}")

        # ── Periodic evaluation vs Alternator ─────────────────────────────────
        if (episode + 1) % EVAL_EVERY == 0:
            coord = evaluate(model, tokenizer, device)
            print(f"\n  ── Eval at episode {episode+1} ──")
            print(f"     Coordination vs Alternator: {coord:.1%}  "
                  f"(paper GPT-4 baseline: ~50%)\n")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n── Final Evaluation ──")
    final_coord = evaluate(model, tokenizer, device, n_episodes=50)
    print(f"  Final coordination rate vs Alternator: {final_coord:.1%}")
    print(f"  Paper's GPT-4 baseline:                ~50%")
    print(f"  Improvement:                           {final_coord - 0.5:+.1%}")

    # ── Save the trained model ────────────────────────────────────────────────
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}/")


if __name__ == "__main__":
    main()
