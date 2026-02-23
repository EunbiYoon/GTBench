# eval.py
# Evaluation utilities for:
#  1) Policy vs TinyLlama (Ollama) opponent
#  2) Approximate "regret" vs TinyLlama using root-only best first move search
#
# Assumptions:
# - Your env class (e.g., TicTacToe4x4Env) provides:
#     reset(starting_player=+1 or -1) -> observation (np.ndarray)
#     legal_actions() -> List[int]
#     step(action) -> StepResult(observation, reward, done, info)
#     render(as_matrix=True) -> str
#     current_player: +1 for X, -1 for O
# - StepResult.info contains "winner" in {+1, -1, 0} at terminal (recommended).
#
# TinyLlama opponent should be implemented in opponent/tiny_llm.py with:
#   OllamaConfig, OllamaLLMOpponent (method: select_action(env) -> int)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type
import random
import numpy as np
import torch

from agents import masked_softmax

# ---- TinyLlama opponent import ----
from opponent.tiny_ollama import OllamaConfig, OllamaLLMOpponent

# -----------------------------
# Policy action selection
# -----------------------------
def policy_select_action(
    model,
    obs: np.ndarray,
    legal_actions: List[int],
    device: str,
    act_dim: int,
    sample: bool = False,
) -> int:
    """
    Pick an action using the PPO policy with legal-action masking.
    sample=False => greedy (argmax)
    """
    if not legal_actions:
        raise RuntimeError("No legal actions.")

    obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # [1, obs_dim]
    mask = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
    mask[0, torch.tensor(legal_actions, dtype=torch.long, device=device)] = 1.0

    with torch.no_grad():
        logits, _ = model(obs_t)
        probs = masked_softmax(logits, mask)  # [1, act_dim]

    if sample:
        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())
    return int(torch.argmax(probs, dim=-1).item())


# -----------------------------
# Match / Episode runner
# -----------------------------
@dataclass
class MatchStats:
    win: int
    loss: int
    draw: int

    @property
    def n(self) -> int:
        return self.win + self.loss + self.draw

    def as_rates(self) -> Dict[str, float]:
        n = max(1, self.n)
        return {
            "win_rate": self.win / n,
            "loss_rate": self.loss / n,
            "draw_rate": self.draw / n,
        }


def _winner_from_terminal(step_info: Dict) -> int:
    """
    Extract winner sign from terminal info. Falls back to 0 if missing.
    """
    w = step_info.get("winner", 0)
    try:
        return int(w)
    except Exception:
        return 0


def play_episode_policy_vs_llm(
    env,
    model,
    device: str,
    act_dim: int,
    llm_opponent,
    policy_plays_as: int,
    policy_sample: bool = False,
    max_plies: int = 16,
) -> int:
    """
    Plays one episode: policy vs LLM.
    Returns winner sign: +1 (X), -1 (O), 0 (draw).

    policy_plays_as: +1 means policy controls X, -1 means policy controls O.
    """
    assert policy_plays_as in (+1, -1)

    # X always starts; environment controls actual turn order via current_player
    obs = env.reset(starting_player=+1)

    for _ in range(max_plies):
        legal = env.legal_actions()
        if not legal:
            # if env ends without done flagged, treat as draw
            return int(getattr(env, "winner", 0))

        if env.current_player == policy_plays_as:
            a = policy_select_action(model, obs, legal, device, act_dim, sample=policy_sample)
        else:
            a = llm_opponent.select_action(env)
            if a not in legal:
                a = random.choice(legal)

        step = env.step(a)
        obs = step.observation

        if step.done:
            return _winner_from_terminal(step.info)

    # horizon reached
    return int(getattr(env, "winner", 0))


# -----------------------------
# Evaluation vs TinyLlama
# -----------------------------
def eval_vs_tinyllama(
    env_cls: Type,
    model,
    device: str,
    act_dim: int,
    episodes: int = 50,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "tinyllama:latest",
    temperature: float = 0.0,
    timeout_sec: float = 10.0,
    policy_sample: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate policy vs TinyLlama (Ollama) opponent for both seats:
      - policy as X (+1)
      - policy as O (-1)

    Returns dict:
      {
        "as_X": { "win_rate": ..., "loss_rate": ..., "draw_rate": ..., "avg_outcome": ... },
        "as_O": { ... }
      }

    avg_outcome is measured from policy perspective (+1 win, -1 loss, 0 draw).
    """
    if OllamaConfig is None or OllamaLLMOpponent is None:
        raise ImportError(
            "Could not import opponent/tiny_llm.py. "
            "Make sure you have opponent/tiny_llm.py with OllamaConfig and OllamaLLMOpponent."
        )

    cfg = OllamaConfig(
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=temperature,
        timeout_sec=timeout_sec,
    )
    llm = OllamaLLMOpponent(cfg)

    def run(policy_seat: int) -> Tuple[MatchStats, float]:
        stats = MatchStats(win=0, loss=0, draw=0)
        outcomes = []

        for _ in range(episodes):
            env = env_cls()
            winner = play_episode_policy_vs_llm(
                env=env,
                model=model,
                device=device,
                act_dim=act_dim,
                llm_opponent=llm,
                policy_plays_as=policy_seat,
                policy_sample=policy_sample,
                max_plies=act_dim,  # max moves
            )

            # convert winner sign into policy-perspective outcome
            if winner == 0:
                stats.draw += 1
                outcomes.append(0)
            elif winner == policy_seat:
                stats.win += 1
                outcomes.append(+1)
            else:
                stats.loss += 1
                outcomes.append(-1)

        return stats, float(np.mean(outcomes))

    stats_x, avg_x = run(policy_seat=+1)
    stats_o, avg_o = run(policy_seat=-1)

    out = {
        "as_X": {**stats_x.as_rates(), "avg_outcome": avg_x, "episodes": stats_x.n},
        "as_O": {**stats_o.as_rates(), "avg_outcome": avg_o, "episodes": stats_o.n},
    }
    return out


# -----------------------------
# Approximate regret vs TinyLlama (root-only)
# -----------------------------
def approx_regret_root_only_vs_tinyllama(
    env_cls: Type,
    model,
    device: str,
    act_dim: int,
    policy_plays_as: int,
    sims_per_action: int = 30,
    sims_baseline: int = 60,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "tinyllama:latest",
    temperature: float = 0.0,
    timeout_sec: float = 10.0,
    policy_sample: bool = False,
) -> float:
    """
    Approx regret (root-only):
      regret = V(best first move) - V(policy's first move)

    Where V(.) is expected outcome from *policy perspective*:
      +1 = policy wins, -1 = policy loses, 0 = draw

    Notes:
    - This uses TinyLlama as the opponent for non-policy turns.
    - This is NOT full best-response regret; only optimizes the first move.
    """
    assert policy_plays_as in (+1, -1)

    if OllamaConfig is None or OllamaLLMOpponent is None:
        raise ImportError(
            "Could not import opponent/tiny_llm.py. "
            "Make sure you have opponent/tiny_llm.py with OllamaConfig and OllamaLLMOpponent."
        )

    cfg = OllamaConfig(
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=temperature,
        timeout_sec=timeout_sec,
    )
    llm = OllamaLLMOpponent(cfg)

    # Root state: X starts
    env0 = env_cls()
    obs0 = env0.reset(starting_player=+1)
    legal0 = env0.legal_actions()
    if not legal0:
        return 0.0

    # --- baseline: policy chooses first move (according to its policy), then play out ---
    base_outcomes = []
    for _ in range(sims_baseline):
        env = env_cls()
        obs = env.reset(starting_player=+1)

        # first move depends on whose turn it is
        legal = env.legal_actions()
        if env.current_player == policy_plays_as:
            a0 = policy_select_action(model, obs, legal, device, act_dim, sample=policy_sample)
        else:
            a0 = llm.select_action(env)
            if a0 not in legal:
                a0 = random.choice(legal)

        step = env.step(a0)
        obs = step.observation

        if step.done:
            winner = _winner_from_terminal(step.info)
        else:
            winner = _play_to_end_with_fixed_players(
                env=env,
                obs=obs,
                model=model,
                device=device,
                act_dim=act_dim,
                llm=llm,
                policy_plays_as=policy_plays_as,
                policy_sample=policy_sample,
            )

        base_outcomes.append(_winner_to_policy_outcome(winner, policy_plays_as))

    v_policy = float(np.mean(base_outcomes))

    # --- search best first move (only if policy moves first at root) ---
    # If policy is not the root player (i.e., policy plays O), then root move belongs to LLM,
    # so root-only best move doesn't make much sense; we still compute by "forcing" LLM's first move
    # which is not controllable. In that case, we return 0.0 or you can define a different regret.
    if policy_plays_as != +1:
        return 0.0

    best_v = -1e9
    for a in legal0:
        vals = []
        for _ in range(sims_per_action):
            env = env_cls()
            obs = env.reset(starting_player=+1)

            # force first move to a (policy controls X at root)
            step = env.step(a)
            obs = step.observation

            if step.done:
                winner = _winner_from_terminal(step.info)
            else:
                winner = _play_to_end_with_fixed_players(
                    env=env,
                    obs=obs,
                    model=model,
                    device=device,
                    act_dim=act_dim,
                    llm=llm,
                    policy_plays_as=policy_plays_as,
                    policy_sample=policy_sample,
                )

            vals.append(_winner_to_policy_outcome(winner, policy_plays_as))

        v = float(np.mean(vals))
        if v > best_v:
            best_v = v

    return max(0.0, best_v - v_policy)


def _winner_to_policy_outcome(winner_sign: int, policy_seat: int) -> int:
    if winner_sign == 0:
        return 0
    return +1 if winner_sign == policy_seat else -1


def _play_to_end_with_fixed_players(
    env,
    obs: np.ndarray,
    model,
    device: str,
    act_dim: int,
    llm,
    policy_plays_as: int,
    policy_sample: bool,
) -> int:
    """
    Continue playing until terminal using:
      - policy on policy_plays_as turns
      - llm otherwise
    Returns winner sign (+1/-1/0).
    """
    for _ in range(act_dim):
        legal = env.legal_actions()
        if not legal:
            return int(getattr(env, "winner", 0))

        if env.current_player == policy_plays_as:
            a = policy_select_action(model, obs, legal, device, act_dim, sample=policy_sample)
        else:
            a = llm.select_action(env)
            if a not in legal:
                a = random.choice(legal)

        step = env.step(a)
        obs = step.observation
        if step.done:
            return _winner_from_terminal(step.info)

    return int(getattr(env, "winner", 0))


# -----------------------------
# Convenience "one-call" eval
# -----------------------------
def eval_reward_and_regret_vs_tinyllama(
    env_cls: Type,
    model,
    device: str,
    act_dim: int,
    episodes: int = 50,
    sims_per_action: int = 30,
    sims_baseline: int = 60,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "tinyllama:latest",
    temperature: float = 0.0,
    timeout_sec: float = 10.0,
    policy_sample: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Returns (reward_p0, reward_p1, regret_p0, regret_p1) where:
      - reward_p0: avg outcome from policy perspective when policy plays X (+1)
      - reward_p1: avg outcome from policy perspective when policy plays O (-1)
      - regret_p0: root-only approx regret when policy plays X
      - regret_p1: root-only approx regret when policy plays O (returns 0.0 by definition here)

    reward is policy perspective: +1 win, -1 loss, 0 draw
    """
    res = eval_vs_tinyllama(
        env_cls=env_cls,
        model=model,
        device=device,
        act_dim=act_dim,
        episodes=episodes,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        temperature=temperature,
        timeout_sec=timeout_sec,
        policy_sample=policy_sample,
    )

    reward_p0 = float(res["as_X"]["avg_outcome"])
    reward_p1 = float(res["as_O"]["avg_outcome"])

    regret_p0 = approx_regret_root_only_vs_tinyllama(
        env_cls=env_cls,
        model=model,
        device=device,
        act_dim=act_dim,
        policy_plays_as=+1,
        sims_per_action=sims_per_action,
        sims_baseline=sims_baseline,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        temperature=temperature,
        timeout_sec=timeout_sec,
        policy_sample=policy_sample,
    )

    regret_p1 = approx_regret_root_only_vs_tinyllama(
        env_cls=env_cls,
        model=model,
        device=device,
        act_dim=act_dim,
        policy_plays_as=-1,
        sims_per_action=sims_per_action,
        sims_baseline=sims_baseline,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        temperature=temperature,
        timeout_sec=timeout_sec,
        policy_sample=policy_sample,
    )

    return reward_p0, reward_p1, float(regret_p0), float(regret_p1)