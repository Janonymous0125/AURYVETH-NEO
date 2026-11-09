"""Normalization, SwiGLU, and causal attention used by the integrated model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FP16/BF16 need FP32 accumulation; FP64 must keep its precision so
        # numerical gradients and high-precision experiments remain correct.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class SwiGLUMLP(nn.Module):
    """Dense gated feed-forward network: down(SiLU(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    """Grouped-query causal attention with rotary position embeddings.

    Only bounded full-sequence attention is used by the current model. Long-term
    neural memory is read separately before this operation in every block.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        kv_width = cfg.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.rope_theta = cfg.rope_theta

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor):
        device = q.device
        inverse_frequencies = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(q.size(-2), device=device, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies).repeat_interleave(2, dim=-1)
        cos = angles.cos()[None, None].to(q.dtype)
        sin = angles.sin()[None, None].to(q.dtype)
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        queries = self.q_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        queries, keys = self._apply_rope(queries, keys)
        # Newer PyTorch kernels support GQA without explicit KV repetition.
        # Keep the original fallback for older supported Torch releases.
        try:
            output = F.scaled_dot_product_attention(
                queries, keys, values,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=(self.n_kv_heads != self.n_heads),
            )
        except TypeError:
            if self.n_kv_heads != self.n_heads:
                groups = self.n_heads // self.n_kv_heads
                keys = keys.repeat_interleave(groups, dim=1)
                values = values.repeat_interleave(groups, dim=1)
            output = F.scaled_dot_product_attention(
                queries, keys, values,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        output = output.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.o_proj(output)
