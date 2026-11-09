"""v0.6 reference attention-free recurrent blocks.

This is a learned gated diagonal recurrent mixer, NOT an implementation of
RWKV-7/Mamba. It provides streaming O(width) state per layer and no self-
attention. The Python reference scan is not a fused high-throughput GPU kernel.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .layers import RMSNorm, SwiGLUMLP
from .memory import NeuralMemoryBank
from .planner import LatentPlanner


class GatedRecurrentMixer(nn.Module):
    """Learnable diagonal-state recurrence, with an O(D) streaming cache.

    s_t = a_t*s_{t-1} + (1-a_t)*tanh(u_t)
    y_t = W_o(silu(s_t) * sigmoid(g_t))

    Gates depend on current input. No future tokens are inspected.
    """

    def __init__(self, width: int):
        super().__init__()
        self.input_proj = nn.Linear(width, width * 3, bias=False)
        self.decay_bias = nn.Parameter(torch.full((width,), 2.0))
        self.output_proj = nn.Linear(width, width, bias=False)
        # Start as a small perturbation; retain nonzero gradients for output_proj.
        self.rescale = nn.Parameter(torch.tensor(0.1))

    def step(self, x: torch.Tensor, state: torch.Tensor | None = None):
        if x.ndim != 2:
            raise ValueError('A recurrent step expects [batch,width]')
        decay, candidate, output_gate = self.input_proj(x).chunk(3, dim=-1)
        decay = torch.sigmoid(self.decay_bias + decay)
        if state is None:
            state = torch.zeros_like(candidate)
        new_state = decay * state + (1 - decay) * torch.tanh(candidate)
        output = self.output_proj(F.silu(new_state) * torch.sigmoid(output_gate))
        return self.rescale * output, new_state

    def forward_reference(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """Original serial recurrence retained for numerical/regression tests."""
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError('Recurrent mixer expects [batch,nonempty tokens,width]')
        results = []
        for token in x.unbind(dim=1):
            y, state = self.step(token, state)
            results.append(y)
        return torch.stack(results, dim=1), state

    @staticmethod
    def _parallel_affine_scan(a: torch.Tensor, b: torch.Tensor):
        """Associative prefix scan for s[t] = a[t]*s[t-1] + b[t].

        Vectorized across the entire sequence; only O(log T) Python-level
        iterations, and no division by decaying cumulative products. This
        PyTorch-only reference is differentiable and needs no CUDA extension.
        """
        count = a.shape[1]
        shift = 1
        while shift < count:
            b = torch.cat((b[:, :shift], b[:, shift:] + a[:, shift:] * b[:, :-shift]), dim=1)
            a = torch.cat((a[:, :shift], a[:, shift:] * a[:, :-shift]), dim=1)
            shift *= 2
        return a, b

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """Vectorized parallel prefix recurrence, equivalent to `step`.

        Chunking bounds the O(log chunk) scan's activation memory. No per-token
        Python loop remains; serial `step` stays as the incremental decoder.
        """
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError('Recurrent mixer expects [batch,nonempty tokens,width]')
        if x.shape[1] <= 4:
            return self.forward_reference(x, state)
        decay, candidate, output_gate = self.input_proj(x).chunk(3, dim=-1)
        coeff = torch.sigmoid(self.decay_bias + decay)
        bias = (1.0 - coeff) * torch.tanh(candidate)
        chunks = []
        # Limit intermediate tensors while batching a 64-token scan.
        for offset in range(0, x.shape[1], 64):
            a, b = self._parallel_affine_scan(coeff[:, offset:offset + 64], bias[:, offset:offset + 64])
            if state is not None:
                states = b + a * state[:, None, :]
            else:
                states = b
            state = states[:, -1, :]
            chunks.append(states)
        full_state = torch.cat(chunks, dim=1)
        out = self.output_proj(F.silu(full_state) * torch.sigmoid(output_gate))
        return self.rescale * out, state


class GatedMemoryRead(nn.Module):
    """Non-attention evidence fusion for the legacy bounded neural memory bank.

    Candidate selection uses an index/cosine similarity, but selected states
    are combined by masked mean, not learned query-key softmax attention.
    """

    def __init__(self, config):
        super().__init__()
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, memory: NeuralMemoryBank, support=None, *, return_state=False):
        states, mask = memory.select(x, support=support)
        if states is None:
            return torch.zeros_like(x)
        count = mask.sum(dim=(-1, -2)).clamp(min=1)
        pooled = (states * mask[..., None]).sum(dim=(2, 3)) / count[..., None]
        has_values = (mask.any(dim=(-1, -2)))[..., None]
        return has_values * torch.sigmoid(self.gate(x)) * self.output(self.value(pooled))


class RecurrentBlock(nn.Module):
    """Neural memory + recurrent mixing + optional planner + SwiGLU."""

    def __init__(self, config):
        super().__init__()
        self.memory_norm = RMSNorm(config.d_model)
        self.memory_read = GatedMemoryRead(config)
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer = GatedRecurrentMixer(config.d_model)
        self.planner = LatentPlanner(config.d_model, config.planner_steps) if config.planner_steps else None
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, memory: NeuralMemoryBank, support=None, *, return_state=False, initial_state=None):
        # Chunk the external-memory lookup to cap temporary allocation.
        if x.shape[1] > 128:
            reads = [self.memory_read(self.memory_norm(x[:, i:i+128]), memory, support)
                     for i in range(0, x.shape[1], 128)]
            delta = torch.cat(reads, dim=1)
        else:
            delta = self.memory_read(self.memory_norm(x), memory, support)
        x = x + self.dropout(delta)
        y, final_state = self.mixer(self.mixer_norm(x), initial_state)
        x = x + self.dropout(y)
        if self.planner is not None:
            x = self.planner(x)
        output = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return (output, final_state) if return_state else output

    def step(self, x: torch.Tensor, memory: NeuralMemoryBank, state=None):
        """Incremental inference; x [B,1,D]; returns (new x, state)."""
        x = x + self.memory_read(self.memory_norm(x), memory)
        y, state = self.mixer.step(self.mixer_norm(x[:, 0]), state)
        x = x + y[:, None, :]
        if self.planner is not None:
            x = self.planner(x)
        x = x + self.mlp(self.mlp_norm(x))
        return x, state
