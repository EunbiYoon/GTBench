# eval.py

import numpy as np
import torch
import pyspiel

from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability

from agents import masked_softmax, ActorCritic


class TorchPolicy(policy_lib.Policy):
    """Wrap your Torch ActorCritic into an OpenSpiel Policy interface."""
    def __init__(self, game, model: ActorCritic, device: str):
        super().__init__(game, list(range(game.num_players())))
        self._model = model
        self._device = device
        self._act_dim = game.num_distinct_actions()

    def action_probabilities(self, state, player_id=None):
        if state.is_chance_node():
            return dict(state.chance_outcomes())

        if player_id is None:
            player_id = state.current_player()

        legal = state.legal_actions(player_id)
        if not legal:
            return {}

        obs = np.asarray(state.information_state_tensor(player_id), dtype=np.float32).reshape(-1)
        mask = np.zeros((self._act_dim,), dtype=np.float32)
        for a in legal:
            if 0 <= a < self._act_dim:
                mask[a] = 1.0

        obs_t = torch.from_numpy(obs).to(self._device).unsqueeze(0)
        mask_t = torch.from_numpy(mask).to(self._device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self._model(obs_t)
            probs = masked_softmax(logits, mask_t).squeeze(0)

        # normalize on legal actions
        out = {a: float(probs[a].item()) for a in legal}
        s = sum(out.values())
        if s <= 0:
            uni = 1.0 / len(legal)
            return {a: uni for a in legal}
        return {a: p / s for a, p in out.items()}


class TwoPlayerJointPolicy(policy_lib.Policy):
    """Joint policy where each player can have a different underlying policy."""
    def __init__(self, game, p0_policy, p1_policy):
        super().__init__(game, list(range(game.num_players())))
        self._p = {0: p0_policy, 1: p1_policy}

    def action_probabilities(self, state, player_id=None):
        if state.is_chance_node():
            return dict(state.chance_outcomes())
        if player_id is None:
            player_id = state.current_player()
        return self._p[player_id].action_probabilities(state, player_id)


def eval_reward_and_regret_vs_fixed(game_name: str, model: ActorCritic, cfr_policy, device: str):
    """
    Returns:
      reward_p0: E[return for PPO] when PPO is player 0 vs CFR player 1
      reward_p1: E[return for PPO] when PPO is player 1 vs CFR player 0
      regret_p0: best_response_value - on_policy_value for PPO (player 0) vs CFR
      regret_p1: best_response_value - on_policy_value for PPO (player 1) vs CFR
    """
    game = pyspiel.load_game(game_name)
    root = game.new_initial_state()

    ppo_policy = TorchPolicy(game, model, device)

    # Seating 1: PPO as P0, CFR as P1
    joint_01 = TwoPlayerJointPolicy(game, ppo_policy, cfr_policy)
    reward_p0 = expected_game_score.policy_value(root, [ppo_policy, cfr_policy])[0]
    br_info_p0 = exploitability.best_response(game, joint_01, player_id=0)
    regret_p0 = br_info_p0["nash_conv"]  # == best_response_value - on_policy_value

    # Seating 2: CFR as P0, PPO as P1
    joint_10 = TwoPlayerJointPolicy(game, cfr_policy, ppo_policy)
    reward_p1 = expected_game_score.policy_value(root, [cfr_policy, ppo_policy])[1]
    br_info_p1 = exploitability.best_response(game, joint_10, player_id=1)
    regret_p1 = br_info_p1["nash_conv"]

    return float(reward_p0), float(reward_p1), float(regret_p0), float(regret_p1)