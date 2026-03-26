"""
Scoring module for optional validation.
Scores individual round reasoning using an LLM judge on 0-100 scale.
Used to verify that level_2 responses consistently outscore level_0.
"""

import time
import json
from openai import OpenAI
from config import (
    LITELLM_BASE_URL, LITELLM_API_KEY, JUDGE_MODEL,
    REQUEST_DELAY_SECONDS
)


JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of strategic reasoning in game theory.

You will be given a player's reasoning for a SINGLE ROUND of a repeated Battle of the Sexes game. In this game:
- Player 1 prefers Opera, Player 2 prefers Football
- (Opera, Opera) → (3,2), (Football, Football) → (2,3), miscoordination → (0,0)
- The key strategic challenge is COORDINATION: both players want to avoid (0,0), but disagree on which coordinated outcome is better.

The player is Player 1 (prefers Opera). They are playing against opponent type: {opponent_type}.
This is round {round_num} of {num_rounds}.

Evaluate the player's reasoning on a scale of 0 to 100 based on these criteria:

1. OPPONENT MODELING (0-25 points): Does the player try to identify what the opponent is doing? Do they recognize patterns? Do they form a model of the opponent's strategy?

2. ADAPTATION (0-25 points): Does the player adjust their strategy based on observed behavior? Do they exploit patterns they've identified?

3. EQUILIBRIUM AWARENESS (0-25 points): Does the player understand that coordination is the goal? Do they recognize the tradeoff between their preferred outcome and successful coordination?

4. COHERENCE (0-25 points): Is the reasoning logically consistent? Does the chosen action follow from the analysis?

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"opponent_modeling": <0-25>, "adaptation": <0-25>, "equilibrium_awareness": <0-25>, "coherence": <0-25>, "total": <0-100>, "brief_justification": "<1-2 sentences>"}}
"""


def score_round_reasoning(
    reasoning: str,
    action: str,
    dpo_prompt: str,
    opponent_type: str,
    round_num: int,
    num_rounds: int = 5,
) -> dict:
    """
    Score a single round's reasoning using the LLM judge.
    Returns dict with sub-scores and total (0-100).
    """
    client = OpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
    )

    system = JUDGE_SYSTEM_PROMPT.format(
        opponent_type=opponent_type,
        round_num=round_num,
        num_rounds=num_rounds,
    )

    user_content = (
        f"Game state:\n{dpo_prompt}\n\n"
        f"Player's reasoning:\n{reasoning}\n\n"
        f"Player's action: {action}"
    )

    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        return _parse_judge_response(text)
    except Exception as e:
        print(f"  [Judge] API error: {e}. Returning default score.")
        return _default_score(str(e))


def validate_game(game_data: dict) -> dict:
    """
    Run validation scoring on a single game's DPO rounds.
    Scores both level_2 (chosen) and level_0 (rejected) reasoning at each round.

    Returns:
        {
            "game_id": str,
            "opponent_type": str,
            "rounds": [
                {
                    "round_num": int,
                    "chosen_score": dict,
                    "rejected_score": dict,
                    "chosen_wins": bool,
                },
                ...
            ],
            "chosen_avg_score": float,
            "rejected_avg_score": float,
            "chosen_win_rate": float,
        }
    """
    results = []
    opp_type = game_data["opponent_type"]

    for round_data in game_data["rounds"]:
        rn = round_data["round_num"]
        prompt = round_data["dpo_prompt"]

        print(f"      Validating round {rn}...")

        chosen_score = score_round_reasoning(
            reasoning=round_data["chosen"]["reasoning"],
            action=round_data["chosen"]["action"],
            dpo_prompt=prompt,
            opponent_type=opp_type,
            round_num=rn,
        )

        rejected_score = score_round_reasoning(
            reasoning=round_data["rejected"]["reasoning"],
            action=round_data["rejected"]["action"],
            dpo_prompt=prompt,
            opponent_type=opp_type,
            round_num=rn,
        )

        results.append({
            "round_num": rn,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "chosen_wins": chosen_score["total"] > rejected_score["total"],
        })

    chosen_scores = [r["chosen_score"]["total"] for r in results]
    rejected_scores = [r["rejected_score"]["total"] for r in results]
    wins = sum(1 for r in results if r["chosen_wins"])

    return {
        "game_id": game_data["game_id"],
        "opponent_type": opp_type,
        "rounds": results,
        "chosen_avg_score": sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0,
        "rejected_avg_score": sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0,
        "chosen_win_rate": wins / len(results) if results else 0,
    }


def _parse_judge_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        expected_keys = ["opponent_modeling", "adaptation", "equilibrium_awareness", "coherence", "total"]
        for key in expected_keys:
            if key not in result:
                raise ValueError(f"Missing key: {key}")
        result["total"] = max(0, min(100, result["total"]))
        return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [Judge] Parse error: {e}. Text: {text[:200]}")
        return _default_score(str(e))


def _default_score(reason: str = "") -> dict:
    return {
        "opponent_modeling": 12,
        "adaptation": 12,
        "equilibrium_awareness": 13,
        "coherence": 13,
        "total": 50,
        "brief_justification": f"Default score: {reason}",
    }
