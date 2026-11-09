"""Bounded differentiable latent refinement within a causal processing block.

This is NOT an autonomous agent or an established reasoning engine. The input
at each position is the causally computed prefix representation; refinement
is position-wise, so no future tokens are visible to earlier positions.
"""
from __future__ import annotations

import torch
from torch import nn
from .layers import RMSNorm


class LatentPlanner(nn.Module):
    """Iteratively update an internal plan hypothesis and gate it into the LM.

    The zero-initialized residual output starts as an identity transform. The
    planner is trained through the main next-token / supervised-answer loss;
    step count is a compute budget, not a claim of correct intermediate plans.
    """

    def __init__(self, width: int, steps: int):
        super().__init__()
        if width < 2 or not 1 <= steps <= 8:
            raise ValueError('Invalid latent planner configuration')
        self.steps = steps
        self.norm = RMSNorm(width)
        self.cell = nn.GRUCell(width, width)
        self.gate = nn.Linear(width * 2, width)
        self.out = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('Planner expects [batch, tokens, width]')
        batch, tokens, width = x.shape
        context = self.norm(x).reshape(batch * tokens, width)
        state = torch.zeros_like(context)
        for _ in range(self.steps):
            state = self.cell(context, state)
        fraction = torch.sigmoid(self.gate(torch.cat((context, state), dim=-1)))
        return x + self.out(state * fraction).reshape(batch, tokens, width)
