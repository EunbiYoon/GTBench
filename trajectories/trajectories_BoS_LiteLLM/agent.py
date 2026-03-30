"""
Agent (Player 1) that uses an LLM to play Battle of the Sexes.

Two modes:
  - play_round(): Used by level_2 to play a live game (builds history incrementally)
  - hypothetical_round(): Used by level_0 to respond to an arbitrary history (no live game)
"""

import time
import random
import re
from openai import OpenAI
from config import (
    LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL,
    PAYOFFS, NUM_ROUNDS, REQUEST_DELAY_SECONDS
)


AGENT_PROMPTS = {
    "level_0": """\
You are Player 1 in a {num_rounds}-round repeated Battle of the Sexes game. You prefer Opera.

Payoff matrix:
  Both choose Opera: you get 3, opponent gets 2
  You: Opera, Opponent: Football: both get 0
  You: Football, Opponent: Opera: both get 0
  Both choose Football: you get 2, opponent gets 3

You want to maximize your payoff. Look at what happened in previous rounds and \
make a quick gut decision. Don't analyze too deeply — just pick what feels right \
based on the immediate situation. Keep it simple.

IMPORTANT: State your brief reasoning (1-2 sentences max), then your final line \
MUST be exactly one of:
Action: Opera
Action: Football
""",

    "level_2": """\
You are Player 1 in a {num_rounds}-round repeated Battle of the Sexes game. You prefer Opera.

Payoff matrix:
  Both choose Opera: you get 3, opponent gets 2
  You: Opera, Opponent: Football: both get 0
  You: Football, Opponent: Opera: both get 0
  Both choose Football: you get 2, opponent gets 3

You are playing {num_rounds} rounds total. You want to maximize your TOTAL payoff \
across all rounds.

Before choosing your action each round, reason carefully about:
1. What pattern do you see in your opponent's past behavior (if any)?
2. What type of strategy might your opponent be using?
3. What do you think your opponent expects YOU to do?
4. Given all this, what is your best action this round?

IMPORTANT: After your reasoning, your final line MUST be exactly one of:
Action: Opera
Action: Football
""",
}


def _build_user_prompt_from_history(history: list[dict], round_num: int, level: str) -> str:
    """Build the user prompt from a history of rounds."""
    prompt = f"Round {round_num} of {NUM_ROUNDS}.\n\n"

    if history:
        prompt += "History of previous rounds:\n"
        for i, round_data in enumerate(history):
            p1 = round_data["agent"]
            p2 = round_data["opponent"]
            payoff = PAYOFFS[(p1, p2)]
            prompt += (
                f"  Round {i + 1}: You played {p1}, Opponent played {p2}. "
                f"Your payoff: {payoff[0]}, Opponent payoff: {payoff[1]}.\n"
            )
        prompt += "\n"
    else:
        prompt += "This is the first round. No history yet.\n\n"

    if level == "level_0":
        prompt += "Pick your action for this round."
    else:
        prompt += (
            "Think step by step about what your opponent is likely to do, "
            "then choose your action."
        )
    return prompt


def _build_dpo_prompt(history: list[dict], round_num: int) -> str:
    """
    Build the structured XML prompt for a DPO pair.
    This is the shared prompt that both chosen and rejected respond to.
    """
    situation = (
        f"You are Player 1 in a {NUM_ROUNDS}-round repeated Battle of the Sexes game. "
        f"You prefer Opera.\n"
        f"Payoff matrix:\n"
        f"  Both choose Opera: you get 3, opponent gets 2\n"
        f"  You: Opera, Opponent: Football: both get 0\n"
        f"  You: Football, Opponent: Opera: both get 0\n"
        f"  Both choose Football: you get 2, opponent gets 3\n"
        f"Your goal is to maximize your total payoff across all {NUM_ROUNDS} rounds.\n"
        f"Before choosing your action, reason about what your opponent is likely to do and why."
    )

    observations = ""
    if history:
        for i, round_data in enumerate(history):
            p1 = round_data["agent"]
            p2 = round_data["opponent"]
            payoff = PAYOFFS[(p1, p2)]
            observations += (
                f"Round {i + 1}: You played {p1}, Opponent played {p2}. "
                f"Your payoff: {payoff[0]}, Opponent payoff: {payoff[1]}.\n"
            )
    else:
        observations = "No previous rounds played yet.\n"

    agent_cumulative = sum(PAYOFFS[(r["agent"], r["opponent"])][0] for r in history)
    opp_cumulative = sum(PAYOFFS[(r["agent"], r["opponent"])][1] for r in history)

    public_info = (
        f"Current round: {round_num} of {NUM_ROUNDS}\n"
        f"Your cumulative payoff so far: {agent_cumulative}\n"
        f"Opponent cumulative payoff so far: {opp_cumulative}"
    )

    private_info = "N/A"

    prompt = (
        f"<situation>\n{situation}\n</situation>\n"
        f"<observations>\n{observations}</observations>\n"
        f"<public_information>\n{public_info}\n</public_information>\n"
        f"<private_information>\n{private_info}\n</private_information>"
    )
    return prompt


def _parse_action(text: str) -> str:
    """Extract the action from the LLM response."""
    lower_text = text.lower()

    # Look for "Action: X" anywhere (with optional markdown bold)
    action_matches = re.findall(
        r'\*{0,2}action\s*:\*{0,2}\s*(opera|football)',
        lower_text
    )
    if action_matches:
        return "Opera" if action_matches[-1] == "opera" else "Football"

    # Look for "I'll play/choose X"
    play_matches = re.findall(
        r"(?:i'?ll\s+(?:play|choose)|my\s+(?:choice|action)\s+is|i\s+(?:play|choose))\s+\*{0,2}(opera|football)",
        lower_text
    )
    if play_matches:
        return "Opera" if play_matches[-1] == "opera" else "Football"

    # Fallback: last mention
    last_opera = lower_text.rfind("opera")
    last_football = lower_text.rfind("football")
    if last_opera > last_football and last_opera != -1:
        return "Opera"
    elif last_football > last_opera and last_football != -1:
        return "Football"

    print(f"  [Agent] Could not parse action. Falling back to random.")
    return random.choice(["Opera", "Football"])


def _extract_reasoning(text: str) -> str:
    """Extract reasoning (everything before the last Action: line)."""
    match = list(re.finditer(r'\*{0,2}action\s*:\*{0,2}\s*(opera|football)', text, re.IGNORECASE))
    if match:
        return text[:match[-1].start()].strip()
    return text.strip()


class Agent:
    def __init__(self, reasoning_level: str = "level_2"):
        if reasoning_level not in AGENT_PROMPTS:
            raise ValueError(f"Unknown level: {reasoning_level}")
        self.reasoning_level = reasoning_level
        self.client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
        )

    def get_response(self, history: list[dict], round_num: int) -> dict:
        """
        Get the agent's reasoning and action for a given game state.
        Works for both live play (level_2) and hypothetical (level_0).

        Returns:
            {"action": str, "reasoning": str, "raw_response": str}
        """
        system = AGENT_PROMPTS[self.reasoning_level].format(num_rounds=NUM_ROUNDS)
        user_prompt = _build_user_prompt_from_history(history, round_num, self.reasoning_level)

        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            response = self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            action = _parse_action(text)
            reasoning = _extract_reasoning(text)
            return {
                "action": action,
                "reasoning": reasoning,
                "raw_response": text,
            }
        except Exception as e:
            print(f"  [Agent-{self.reasoning_level}] API error: {e}. Falling back to random.")
            return {
                "action": random.choice(["Opera", "Football"]),
                "reasoning": f"API error: {e}",
                "raw_response": "",
            }


# Export the prompt builder for use in DPO formatting
build_dpo_prompt = _build_dpo_prompt
