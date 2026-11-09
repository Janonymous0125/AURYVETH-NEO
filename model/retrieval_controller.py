"""Optional supervised, learned retrieval decision head.

Not trained by ordinary LM pretraining. Must be explicitly trained and
validated on labeled question→needs-external-evidence examples before use.
"""
import torch
from torch import nn


class RetrievalController(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self, hidden: torch.Tensor):
        if hidden.ndim != 2:
            raise ValueError('Expected [batch,width] question representations')
        return self.net(hidden).squeeze(-1)
