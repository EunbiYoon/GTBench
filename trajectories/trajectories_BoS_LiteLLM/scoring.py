"""
Scoring module for trajectory evaluation.
Three components: coordination, payoff, and reasoning quality (LLM judge).
Final score is a weighted average on 0-100 scale.
"""

import time
from openai import OpenAI
from config import (
    LITELLM_BASE_URL, LITELLM_API_KEY, JUDGE_MODEL,
    MAX_PAYOFF, MIN_PAYOFF, MAX_COORDINATION, MIN_COORDINATION,
    WEIGHT_COORDINATION, WEIGHT_PAYOFF, WEIGHT_REASONING,
    NUM_ROUNDS, REQUEST_DELAY_SECONDS
)


JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of strategic reasoning in game theory.

You will be given a player's reasoning across multiple rounds of a repeated Battle of the Sexes game. In this game:
- Player 1 prefers Opera, Player 2 prefers Football
- (Opera, Opera) → (3,2), (Football, Football) → (2,3), miscoordination → (0,0)
- The key strategic challenge is COORDINATION: both players want to avoid (0,0), but disagree on which coordinated outcome is better.

The player was Player 1 (prefers Opera). They played against an opponent of type: {opponent_type}.

Evaluate the player's reasoning on a scale of 0 to 100 based on these criteria:

1. OPPONENT MODELING (0-25 points): Does the player try to identify what the opponent is doing? Do they recognize patterns in the opponent's behavior? Do they form a model of the opponent's strategy?

2. ADAPTATION (0-25 points): Does the player adjust their strategy based on observed opponent behavior? Do they change approach when something isn't working? Do they exploit patterns they've identified?

3. EQUILIBRIUM AWARENESS (0-25 points): Does the player understand that coordination is the goal? Do they reason about Nash equilibria, dominant strategies, or coordination incentives? Do they recognize the tradeoff between their preferred outcome (Opera) and successful coordination?

4. COHERENCE (0-25 points): Is the reasoning logically consistent? Does the chosen action follow from the analysis? Is the reasoning specific to the game state rather than generic?

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"opponent_modeling": <0-25>, "adaptation": <0-25>, "equilibrium_awareness": <0-25>, "coherence": <0-25>, "total": <0-100>, "brief_justification": "<1-2 sentences>"}}
"""


def score_coordination(trajectory: dict) -> float:
    """Score coordination on 0-100 scale."""
    coord = trajectory["coordination_count"]
    if MAX_COORDINATION == MIN_COORDINATION:
        return 100.0
    normalized = (coord - MIN_COORDINATION) / (MAX_COORDINATION - MIN_COORDINATION)
    return round(normalized * 100, 2)


def score_payoff(trajectory: dict) -> float:
    """Score agent's total payoff on 0-100 scale."""
    payoff = trajectory["total_agent_payoff"]
    if MAX_PAYOFF == MIN_PAYOFF:
        return 100.0
    normalized = (payoff - MIN_PAYOFF) / (MAX_PAYOFF - MIN_PAYOFF)
    return round(normalized * 100, 2)


def score_reasoning(trajectory: dict) -> dict:
    """
    Use LLM judge to score the agent's reasoning quality on 0-100 scale.
    Returns the full judge response dict including sub-scores.
    """
    client = OpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
    )

    # Build the reasoning text to evaluate
    reasoning_text = _format_trajectory_for_judge(trajectory)
    system = JUDGE_SYSTEM_PROMPT.format(opponent_type=trajectory["opponent_type"])

    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": reasoning_text},
            ],
            max_tokens=300,
            temperature=0.0,  # deterministic judging
        )
        text = response.choices[0].message.content.strip()
        return _parse_judge_response(text)
    except Exception as e:
        print(f"  [Judge] API error: {e}. Returning default score of 50.")
        return {
            "opponent_modeling": 12,
            "adaptation": 12,
            "equilibrium_awareness": 13,
            "coherence": 13,
            "total": 50,
            "brief_justification": f"Judge API error: {e}",
        }


def _format_trajectory_for_judge(trajectory: dict) -> str:
    """Format the full trajectory for the judge to evaluate."""
    lines = [
        f"Game against opponent type: {trajectory['opponent_type']}",
        f"Total rounds: {NUM_ROUNDS}",
        f"Final agent payoff: {trajectory['total_agent_payoff']}",
        f"Coordination count: {trajectory['coordination_count']}/{NUM_ROUNDS}",
        "",
    ]
    for r in trajectory["rounds"]:
        lines.append(f"--- Round {r['round_num']} ---")
        lines.append(f"Agent reasoning: {r['agent_reasoning']}")
        lines.append(f"Agent action: {r['agent_action']}")
        lines.append(f"Opponent action: {r['opponent_action']}")
        lines.append(f"Payoffs: Agent={r['agent_payoff']}, Opponent={r['opponent_payoff']}")
        lines.append("")
    return "\n".join(lines)


def _parse_judge_response(text: str) -> dict:
    """Parse the JSON response from the judge LLM."""
    import json

    # Clean up potential markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        # Validate expected keys
        expected_keys = [
            "opponent_modeling", "adaptation",
            "equilibrium_awareness", "coherence", "total"
        ]
        for key in expected_keys:
            if key not in result:
                raise ValueError(f"Missing key: {key}")
        # Clamp total to 0-100
        result["total"] = max(0, min(100, result["total"]))
        return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [Judge] Could not parse response: {text[:200]}. Error: {e}")
        return {
            "opponent_modeling": 12,
            "adaptation": 12,
            "equilibrium_awareness": 13,
            "coherence": 13,
            "total": 50,
            "brief_justification": f"Parse error: {e}",
        }


def compute_final_score(trajectory: dict) -> dict:
    """
    Compute the final weighted score for a trajectory.
    Returns a dict with all component scores and the final score.
    """
    coord_score = score_coordination(trajectory)
    payoff_score_val = score_payoff(trajectory)
    reasoning_result = score_reasoning(trajectory)
    reasoning_score = reasoning_result["total"]

    final_score = (
        WEIGHT_COORDINATION * coord_score
        + WEIGHT_PAYOFF * payoff_score_val
        + WEIGHT_REASONING * reasoning_score
    )
    final_score = round(final_score, 2)

    return {
        "coordination_score": coord_score,
        "payoff_score": payoff_score_val,
        "reasoning_score": reasoning_score,
        "reasoning_breakdown": reasoning_result,
        "final_score": final_score,
        "weights": {
            "coordination": WEIGHT_COORDINATION,
            "payoff": WEIGHT_PAYOFF,
            "reasoning": WEIGHT_REASONING,
        },
    }
