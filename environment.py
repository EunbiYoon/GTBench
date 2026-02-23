# environment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


def action_to_rc(action: int, n: int = 4) -> Tuple[int, int]:
    return action // n, action % n


def rc_to_action(r: int, c: int, n: int = 4) -> int:
    return r * n + c


@dataclass
class StepResult:
    observation: np.ndarray  # float32 vector
    reward: float            # reward for the player who just moved (current_player before step)
    done: bool
    info: Dict


class TicTacToe4x4Env:
    """
    4x4 TicTacToe (4-in-a-row).
    - Action space: 0..15
    - Board stores move order:
        X_k as +k, O_k as -k
    - current_player: +1 for X, -1 for O
    - step(action): reward is for the player who just moved
        win => +1, draw/ongoing => 0
    - observation: float32 vector length 17:
        16 cells: X_k -> +k/16, O_k -> -k/16, empty -> 0
        1 cell: current_player (+1 or -1)
    """

    def __init__(self):
        self.n = 4
        self.reset(starting_player=1)

    def reset(self, starting_player: int = 1) -> np.ndarray:
        assert starting_player in (+1, -1)
        self.board = np.zeros((self.n, self.n), dtype=np.int32)  # stores +/- order index
        self.current_player = starting_player
        self.x_count = 0
        self.o_count = 0
        self.done = False
        self.winner = 0  # +1 X, -1 O, 0 draw/ongoing
        return self._get_obs()

    def legal_actions(self) -> List[int]:
        if self.done:
            return []
        empties = np.argwhere(self.board == 0)
        return [rc_to_action(int(r), int(c), self.n) for r, c in empties]

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self._get_obs(), 0.0, True, {"winner": self.winner, "reason": "done"})

        r, c = action_to_rc(action, self.n)
        if not (0 <= r < self.n and 0 <= c < self.n):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.n*self.n-1}].")
        if self.board[r, c] != 0:
            raise ValueError(f"Illegal action {action} at (r={r}, c={c}): cell occupied.")

        player = self.current_player  # player who is making the move now

        # place with order index
        if player == 1:
            self.x_count += 1
            self.board[r, c] = +self.x_count
        else:
            self.o_count += 1
            self.board[r, c] = -self.o_count

        # check terminal
        if self._is_win(player):
            self.done = True
            self.winner = player
            reward = 1.0
            info = {"winner": self.winner, "reason": "win", "last_action": action, "last_rc": (r, c)}
            return StepResult(self._get_obs(), reward, True, info)

        if len(self.legal_actions()) == 0:
            self.done = True
            self.winner = 0
            reward = 0.0
            info = {"winner": self.winner, "reason": "draw", "last_action": action, "last_rc": (r, c)}
            return StepResult(self._get_obs(), reward, True, info)

        # switch player
        self.current_player *= -1
        reward = 0.0
        info = {"winner": 0, "reason": "continue", "last_action": action, "last_rc": (r, c)}
        return StepResult(self._get_obs(), reward, False, info)

    def render(self, as_matrix: bool = True) -> str:
        def cell_str(v: int) -> str:
            if v == 0:
                return "."
            if v > 0:
                return f"X_{v}"
            return f"O_{abs(v)}"

        rows = []
        for r in range(self.n):
            row_tokens = [cell_str(int(self.board[r, c])) for c in range(self.n)]
            # align columns nicely
            w = max(3, max(len(t) for t in row_tokens))
            rows.append(" ".join([t.rjust(w) for t in row_tokens]))
        return "\n".join(rows)

    # -----------------------
    # Helpers
    # -----------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n * self.n + 1,), dtype=np.float32)
        flat = self.board.reshape(-1)
        for i, v in enumerate(flat):
            if v == 0:
                obs[i] = 0.0
            elif v > 0:
                obs[i] = float(v) / 16.0
            else:
                obs[i] = -float(abs(v)) / 16.0
        obs[-1] = float(self.current_player)
        return obs

    def _is_win(self, player: int) -> bool:
        assert player in (+1, -1)
        occ = (self.board > 0) if player == 1 else (self.board < 0)

        # rows / cols
        for r in range(self.n):
            if np.all(occ[r, :]):
                return True
        for c in range(self.n):
            if np.all(occ[:, c]):
                return True

        # diagonals
        if np.all(np.diag(occ)):
            return True
        if np.all(np.diag(np.fliplr(occ))):
            return True

        return False