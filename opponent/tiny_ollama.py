# opponent.py
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import List, Optional

import requests


DEFAULT_SYSTEM_PROMPT = """You are playing 4x4 TicTacToe (4-in-a-row).
You must output ONLY one integer action in [0, 15].
Action mapping: row = action // 4, col = action % 4.
Choose an action from the provided legal actions only.
Do NOT output any extra words, punctuation, or explanation—ONLY the number.
"""


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "tinyllama:latest"
    temperature: float = 0.0
    top_p: float = 0.9
    num_predict: int = 16
    timeout_sec: float = 10.0


class OllamaLLMOpponent:
    """
    Uses Ollama local HTTP API to pick an action.
    Works with env that supports:
      - env.render(as_matrix=True) -> str
      - env.legal_actions() -> List[int]
      - env.current_player: +1 for X, -1 for O
    """

    def __init__(self, cfg: OllamaConfig, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.cfg = cfg
        self.system_prompt = system_prompt

    def _call_ollama_generate(self, prompt: str) -> str:
        """
        Calls Ollama /api/generate (simple completion style).
        """
        url = f"{self.cfg.base_url}/api/generate"
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.num_predict,
            },
        }
        r = requests.post(url, json=payload, timeout=self.cfg.timeout_sec)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()

    @staticmethod
    def _extract_action(text: str) -> Optional[int]:
        """
        Extract first integer from model output.
        """
        m = re.search(r"-?\d+", text)
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None

    def select_action(self, env) -> int:
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("No legal actions available.")

        board = env.render(as_matrix=True) if hasattr(env, "render") else str(env)
        player = getattr(env, "current_player", None)

        user_prompt = (
            f"{self.system_prompt}\n\n"
            f"Current player: {'X(+1)' if player == 1 else 'O(-1)' if player == -1 else str(player)}\n"
            f"Board:\n{board}\n\n"
            f"Legal actions: {legal}\n"
            f"Output one action integer now:"
        )

        try:
            out = self._call_ollama_generate(user_prompt)
            a = self._extract_action(out)
            if a is None or a not in legal:
                # fallback: sometimes LLM returns illegal
                return random.choice(legal)
            return a
        except Exception:
            # network/model error fallback
            return random.choice(legal)