# agents.py
# Model + action selection utilities.

import numpy as np
import torch
import torch.nn as nn

from config import HIDDEN


class ActorCritic(nn.Module):
    """
    Simple shared-body actor-critic:
      - Actor outputs logits over all actions.
      - Critic outputs a scalar state-value V(s).
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Softmax that assigns ~0 probability to illegal actions.

    logits: [B, A]
    mask:   [B, A] with 1.0 for legal, 0.0 for illegal
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(
        mask > 0,
        logits,
        torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype),
    )
    return torch.softmax(masked_logits, dim=-1)


def sample_action_from_env_timestep(ts, player_id: int, model: ActorCritic, act_dim: int, device: str) -> int:
    """
    Sample an action for `player_id` from the PPO policy using rl_environment timestep.

    Uses:
      - ts.observations["info_state"][player_id] as model input
      - ts.observations["legal_actions"][player_id] to build a mask
    """
    obs = np.asarray(ts.observations["info_state"][player_id], dtype=np.float32).reshape(-1)
    legal = ts.observations["legal_actions"][player_id]

    # Build action mask (1 for legal, 0 for illegal)
    mask = np.zeros((act_dim,), dtype=np.float32)
    for a in legal:
        if 0 <= a < act_dim:
            mask[a] = 1.0

    obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)     # [1, obs_dim]
    mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)   # [1, act_dim]

    with torch.no_grad():
        logits, _ = model(obs_t)
        probs = masked_softmax(logits, mask_t).squeeze(0)  # [act_dim]

        # Safety normalize / fallback
        s = probs.sum().item()
        if s <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        else:
            probs = probs / probs.sum()

        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())