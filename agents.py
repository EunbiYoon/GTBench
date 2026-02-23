# agents.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-20):
    """
    logits: [B, A]
    mask:   [B, A] with 1.0 for legal actions, 0.0 for illegal
    """
    mask = mask.to(dtype=logits.dtype)
    # Set illegal logits to very negative
    illegal = (mask <= 0)
    masked_logits = logits.masked_fill(illegal, -1e9)
    probs = F.softmax(masked_logits, dim=dim)
    # numerical safety: if all were illegal (shouldn't happen), fallback uniform later
    probs = probs * mask
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs / denom


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value