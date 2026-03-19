"""
Agent (Player 1) that uses an LLM to play Battle of the Sexes.
Uses Option (b): prompt includes game setup + history, agent reasons about opponent.
"""

import time
import random
from openai import OpenAI
from config import (
    LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL,
    PAYOFFS, NUM_ROUNDS, REQUEST_DELAY_SECONDS
)


AGENT_SYSTEM_PROMPT = """\
You are playing a repeated Battle of the Sexes game as Player 1. You prefer Opera.

Payoff matrix:
  If both choose Opera:    you get 3, opponent gets 2
  If you choose Opera, opponent chooses Football:  both get 0
  If you choose Football, opponent chooses Opera:  both get 0
  If both choose Football: you get 2, opponent gets 3

You are playing {num_rounds} rounds total. You want to maximize your TOTAL payoff across all rounds.

Before choosing your action each round, reason carefully about:
1. What pattern do you see in your opponent's past behavior (if any)?
2. What type of strategy might your opponent be using?
3. What do you think your opponent expects YOU to do?
4. Given all this, what is your best action this round?

IMPORTANT: After your reasoning, your final line MUST be exactly one of:
Action: Opera
Action: Football
"""


class Agent:
    def __init__(self):
        self.client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
        )

    def choose_action(self, history: list[dict], round_num: int) -> dict:
        """
        Returns a dict with:
          - "action": "Opera" or "Football"
          - "reasoning": the full chain-of-thought text
          - "raw_response": the complete LLM response
        """
        system = AGENT_SYSTEM_PROMPT.format(num_rounds=NUM_ROUNDS)
        user_prompt = self._build_user_prompt(history, round_num)

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
            action = self._parse_action(text)
            reasoning = self._extract_reasoning(text)

            return {
                "action": action,
                "reasoning": reasoning,
                "raw_response": text,
            }
        except Exception as e:
            print(f"  [Agent] API error: {e}. Falling back to random.")
            return {
                "action": random.choice(["Opera", "Football"]),
                "reasoning": f"API error: {e}",
                "raw_response": "",
            }

    def _build_user_prompt(self, history: list[dict], round_num: int) -> str:
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

        prompt += (
            "Think step by step about what your opponent is likely to do, "
            "then choose your action."
        )
        return prompt

    def _parse_action(self, text: str) -> str:
        """Extract the action from the LLM response."""
        import re
        lower_text = text.lower()

        # Look for "Action: X" anywhere (with optional markdown bold)
        action_matches = re.findall(
            r'\*{0,2}action\s*:\*{0,2}\s*(opera|football)',
            lower_text
        )
        if action_matches:
            return "Opera" if action_matches[-1] == "opera" else "Football"

        # Look for "I'll play/choose X" or "my choice is X"
        play_matches = re.findall(
            r"(?:i'?ll\s+(?:play|choose)|my\s+(?:choice|action)\s+is|i\s+(?:play|choose))\s+\*{0,2}(opera|football)",
            lower_text
        )
        if play_matches:
            return "Opera" if play_matches[-1] == "opera" else "Football"

        # Fallback: look for last mention of opera/football
        last_opera = lower_text.rfind("opera")
        last_football = lower_text.rfind("football")
        if last_opera > last_football and last_opera != -1:
            return "Opera"
        elif last_football > last_opera and last_football != -1:
            return "Football"

        print(f"  [Agent] Could not parse action from response. Falling back to random.")
        return random.choice(["Opera", "Football"])

    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning (everything before the last Action: line)."""
        import re
        # Find the last "Action:" occurrence and take everything before it
        match = list(re.finditer(r'\*{0,2}action\s*:\*{0,2}\s*(opera|football)', text, re.IGNORECASE))
        if match:
            return text[:match[-1].start()].strip()
        # No Action: line found — return the whole text as reasoning
        return text.strip()
