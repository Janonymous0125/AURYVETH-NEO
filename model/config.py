from dataclasses import dataclass
import math
from numbers import Real


@dataclass
class ModelConfig:
    """Architecture parameters. Edit these to change model scale/structure."""

    vocab_size: int = 50_000  # Requires a trained matching tokenizer
    block_size: int = 8192
    d_model: int = 1024
    n_layers: int = 4
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_hidden: int = 3072
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 258:
            raise ValueError("The byte-level tokenizer requires vocab_size >= 258")
        if self.block_size < 2 or self.d_model < 2 or self.n_layers < 1:
            raise ValueError("Invalid model size")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("Attention head counts must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if (not isinstance(self.rope_theta, Real) or isinstance(self.rope_theta, bool)
                or not math.isfinite(self.rope_theta) or self.rope_theta <= 0):
            raise ValueError("rope_theta must be a finite positive real number")
        # RoPE computes frequencies/angles in float32. Even a finite positive
        # base can overflow angles at the configured maximum context length.
        head_dim = self.d_model // self.n_heads
        max_log_angle = math.log(self.block_size - 1)
        if self.rope_theta < 1 and head_dim > 2:
            max_log_angle -= ((head_dim - 2) / head_dim) * math.log(self.rope_theta)
        if max_log_angle >= math.log(3.4028234663852886e38):
            raise ValueError("rope_theta would overflow float32 RoPE angles")
        if self.ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
