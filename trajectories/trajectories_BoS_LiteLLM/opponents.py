"""
Opponent strategies for Battle of the Sexes.
Each opponent is a callable that takes game history and returns an action.
Scripted opponents have epsilon-noise for diversity.
"""

import random
import time
from openai import OpenAI
from config import (
    ACTIONS, EPSILON, LITELLM_BASE_URL, LITELLM_API_KEY,
    LLM_OPPONENT_MODEL, PAYOFFS, REQUEST_DELAY_SECONDS
)


def apply_epsilon_noise(intended_action: str, epsilon: float = EPSILON) -> str:
    """With probability epsilon, play the opposite action."""
    if random.random() < epsilon:
        return "Football" if intended_action == "Opera" else "Opera"
    return intended_action


class AlwaysOpera:
    """Always plays Opera (regardless of history)."""
    name = "AlwaysOpera"

    def choose_action(self, history: list[dict]) -> str:
        return apply_epsilon_noise("Opera")


class AlwaysFootball:
    """Always plays Football (regardless of history)."""
    name = "AlwaysFootball"

    def choose_action(self, history: list[dict]) -> str:
        return apply_epsilon_noise("Football")


class Alternator:
    """Alternates between Opera and Football. Starts with Opera."""
    name = "Alternator"

    def choose_action(self, history: list[dict]) -> str:
        round_num = len(history)
        intended = "Opera" if round_num % 2 == 0 else "Football"
        return apply_epsilon_noise(intended)


class RandomOpponent:
    """Plays uniformly at random each round."""
    name = "Random"

    def choose_action(self, history: list[dict]) -> str:
        # No epsilon noise needed — already random
        return random.choice(ACTIONS)


class ConditionalCooperator:
    """
    Matches the agent's last action. Round 1: plays Opera.
    'Cooperates' by going along with whatever the agent did last.
    """
    name = "ConditionalCooperator"

    def choose_action(self, history: list[dict]) -> str:
        if len(history) == 0:
            return apply_epsilon_noise("Opera")
        last_agent_action = history[-1]["agent"]
        return apply_epsilon_noise(last_agent_action)


class LLMOpponent:
    """
    An LLM playing as Player 2 (prefers Football) via the LiteLLM endpoint.
    """
    name = "LLM"

    def __init__(self):
        self.client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
        )

    def choose_action(self, history: list[dict]) -> str:
        system_prompt = (
            "You are playing a repeated Battle of the Sexes game as Player 2. "
            "You prefer Football.\n\n"
            "Payoff matrix:\n"
            "  (Opera, Opera)    → Player 1 gets 3, you get 2\n"
            "  (Opera, Football) → both get 0\n"
            "  (Football, Opera) → both get 0\n"
            "  (Football, Football) → Player 1 gets 2, you get 3\n\n"
            "You want to maximize YOUR total payoff over all rounds. "
            "Think briefly about what your opponent might do, then choose your action.\n\n"
            "IMPORTANT: Your final line MUST be exactly one of:\n"
            "Action: Opera\n"
            "Action: Football"
        )

        history_text = self._format_history(history)
        user_prompt = f"Round {len(history) + 1} of 5.\n\n"
        if history_text:
            user_prompt += f"History of previous rounds:\n{history_text}\n\n"
        else:
            user_prompt += "This is the first round. No history yet.\n\n"
        user_prompt += "What do you play?"

        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            response = self.client.chat.completions.create(
                model=LLM_OPPONENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            return self._parse_action(text)
        except Exception as e:
            print(f"  [LLMOpponent] API error: {e}. Falling back to random.")
            return random.choice(ACTIONS)

    def _format_history(self, history: list[dict]) -> str:
        lines = []
        for i, round_data in enumerate(history):
            p1 = round_data["agent"]
            p2 = round_data["opponent"]
            payoff = PAYOFFS[(p1, p2)]
            lines.append(
                f"  Round {i + 1}: Player 1 played {p1}, You played {p2}. "
                f"Payoffs: Player 1 got {payoff[0]}, You got {payoff[1]}."
            )
        return "\n".join(lines)

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

        # Fallback: look for the last mention of opera/football in the text
        last_opera = lower_text.rfind("opera")
        last_football = lower_text.rfind("football")
        if last_opera > last_football and last_opera != -1:
            return "Opera"
        elif last_football > last_opera and last_football != -1:
            return "Football"

        print(f"  [LLMOpponent] Could not parse action from: {text[:100]}. Falling back to random.")
        return random.choice(ACTIONS)


def get_opponent(opponent_type: str):
    """Factory function to get an opponent instance by type name."""
    opponents = {
        "AlwaysOpera": AlwaysOpera,
        "AlwaysFootball": AlwaysFootball,
        "Alternator": Alternator,
        "Random": RandomOpponent,
        "ConditionalCooperator": ConditionalCooperator,
        "LLM": LLMOpponent,
    }
    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}. Available: {list(opponents.keys())}")
    return opponents[opponent_type]()
