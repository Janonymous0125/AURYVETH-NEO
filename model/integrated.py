"""Autoregressive language model with persistent, in-network neural memory.

Memory representations are registered buffers rather than learned weights. Each
processing block reads a bounded selection before self-attention and the MLP.
The associative memory graph is metadata, not a neural-network connection.
"""

from __future__ import annotations

import hashlib
import copy
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from neural_tokenizer import LayeredTokenizer, SentencePieceTokenizer
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig
from .layers import CausalSelfAttention, RMSNorm, SwiGLUMLP
from .memory import NeuralMemoryBank, NeuralMemoryRead
from .planner import LatentPlanner
from .recurrent import RecurrentBlock
from .retrieval_controller import RetrievalController
from .evidence import evidence_prompt


@dataclass
class IntegratedConfig(ModelConfig):
    memory_slots: int = 128
    memory_tokens: int = 64
    regions: int = 4
    active_regions: int = 2
    active_memories: int = 4
    planner_steps: int = 0  # 0 retains exact v0.5 architecture and checkpoints
    backbone: str = "transformer"  # "recurrent" opts in to v0.6 attention-free path

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backbone not in ("transformer", "recurrent"):
            raise ValueError("backbone must be transformer or recurrent")
        if not 1 <= self.memory_slots <= 4096 or not 1 <= self.memory_tokens <= 256:
            raise ValueError("Invalid neural memory capacity")
        if not 1 <= self.regions <= 128 or not 1 <= self.active_regions <= self.regions:
            raise ValueError("Invalid memory region configuration")
        if type(self.planner_steps) is not int or not 0 <= self.planner_steps <= 8:
            raise ValueError("planner_steps must be 0..8")
        if not 1 <= self.active_memories <= self.memory_slots:
            raise ValueError("Invalid active memory budget")


@dataclass
class IntegratedOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None = None


class IntegratedBlock(nn.Module):
    """Memory fusion precedes causal self-attention and SwiGLU."""

    def __init__(self, config: IntegratedConfig):
        super().__init__()
        self.memory_norm = RMSNorm(config.d_model)
        self.memory_read = NeuralMemoryRead(config)
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.planner = LatentPlanner(config.d_model, config.planner_steps) if config.planner_steps else None
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, memory: NeuralMemoryBank, support=None) -> torch.Tensor:
        normalized = self.memory_norm(x)
        # Limit peak memory while maintaining exactly the same per-token routing
        # semantics; memory.select is positionwise (not sequence attention).
        if normalized.size(1) > 128:
            reads = [self.memory_read(normalized[:, i:i + 128], memory,
                                      None if support is None else support)
                     for i in range(0, normalized.size(1), 128)]
            memory_delta = torch.cat(reads, dim=1)
        else:
            memory_delta = self.memory_read(normalized, memory, support)
        x = x + self.dropout(memory_delta)
        attention = self.attn(self.attn_norm(x))
        x = x + self.dropout(attention)
        if self.planner is not None:
            x = self.planner(x)
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


class IntegratedLM(nn.Module):
    """Causal language model with persistent internal neural memory buffers."""

    CHECKPOINT_SCHEMA = 2

    def __init__(self, config: IntegratedConfig):
        super().__init__()
        self.config = config
        width = config.d_model
        self.token_embedding = nn.Embedding(config.vocab_size, width)
        # Separate, trainable-before-use memory codec. Never share its weights with the LM.
        self.memory_token_embedding = nn.Embedding(config.vocab_size, width)
        self.memory_encoder = nn.GRU(width, width // 2, bidirectional=True, batch_first=True)
        self.memory_state_proj = nn.Linear(width, width)
        # Versioned candidate-encoder outputs must be mapped to one stable
        # neural memory interface understood by existing processing readers.
        self.register_buffer('codec_to_memory_matrix', torch.eye(width))
        self.register_buffer('codec_to_memory_bias', torch.zeros(width))
        self.memory = NeuralMemoryBank(config)
        block_class = RecurrentBlock if config.backbone == "recurrent" else IntegratedBlock
        self.blocks = nn.ModuleList([block_class(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(width)
        # Only recurrent v0.6 has the optional learned retrieval decision.
        # Remains disabled until explicitly supervised and validated.
        if config.backbone == "recurrent":
            self.retrieval_controller = RetrievalController(width)
            self.register_buffer("retrieval_controller_trained", torch.tensor(False))
        self.lm_head = nn.Linear(width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        # Generic Linear initialization would otherwise erase the planner's
        # identity-preserving zero residual in freshly initialized models.
        for block in self.blocks:
            if block.planner is not None:
                nn.init.zeros_(block.planner.out.weight)
        self._tokenizer_contract: dict | None = None
        self._fingerprint_cache = None
        self.gradient_checkpointing = False
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        # Start the independent codec from the same vocabulary geometry.
        with torch.no_grad():
            self.memory_token_embedding.weight.copy_(self.token_embedding.weight)

    @staticmethod
    def _initialize(layer: nn.Module) -> None:
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            nn.init.normal_(layer.weight, std=0.02)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def add_planner_to_existing_model(self, steps: int) -> None:
        """Explicit identity-preserving extension of a v0.5 model.

        Only supported on a model without planners. The residual projection
        is zero-initialized, so logits/memories remain identical at migration;
        future training can learn the new parameters. Does not alter the
        memory codec, tokenizer contract or persistent memory contents.
        """
        if type(steps) is not int or not 1 <= steps <= 8:
            raise ValueError('Planner steps must be between 1 and 8')
        if self.config.planner_steps:
            raise ValueError('Model already contains planner blocks')
        device = self.token_embedding.weight.device
        dtype = self.token_embedding.weight.dtype
        for block in self.blocks:
            block.planner = LatentPlanner(self.config.d_model, steps).to(device=device, dtype=dtype)
        self.config.planner_steps = steps

    def _check_ids(self, ids: torch.Tensor, *, allow_three_dim: bool = False) -> None:
        allowed = (2, 3) if allow_three_dim else (2,)
        if (
            not isinstance(ids, torch.Tensor)
            or ids.dtype != torch.long
            or ids.ndim not in allowed
            or ids.numel() == 0
            or ids.min() < 0
            or ids.max() >= self.config.vocab_size
        ):
            raise ValueError("Expected nonempty, in-vocabulary int64 token IDs")

    def _encode_support(self, ids: torch.Tensor, lengths: torch.Tensor):
        self._check_ids(ids, allow_three_dim=True)
        if ids.ndim != 3:
            raise ValueError("support_ids must be [B,M,L]")
        batch, memory_count, tokens = ids.shape
        if (
            tokens > self.config.memory_tokens
            or not isinstance(lengths, torch.Tensor)
            or lengths.shape != (batch, memory_count)
            or lengths.dtype != torch.long
            or torch.any(lengths < 1)
            or torch.any(lengths > tokens)
        ):
            raise ValueError("Invalid support lengths or memory size")
        embeddings = self.memory_token_embedding(ids.reshape(batch * memory_count, tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.reshape(-1).cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.memory_encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=tokens)
        states = self.memory_state_proj(output).reshape(batch, memory_count, tokens, -1)
        states = states @ self.codec_to_memory_matrix + self.codec_to_memory_bias
        mask = torch.arange(tokens, device=ids.device)[None, None, :] < lengths[:, :, None]
        states = states * mask[..., None]
        keys = F.normalize(states.sum(2) / lengths[..., None], dim=-1, eps=1e-8)
        return states, lengths, keys

    def _codec_parameters(self):
        return tuple(self.memory_token_embedding.parameters()) + tuple(self.memory_encoder.parameters()) + tuple(self.memory_state_proj.parameters())

    def _freeze_memory_codec(self) -> None:
        """After first persistent write, ordinary training cannot shift stored-memory space."""
        for parameter in self._codec_parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def _encoder_fingerprint(self) -> str:
        # Check PyTorch tensor mutation counters instead of copying and hashing
        # 100+ MB of frozen encoder weights on *every forward call*.
        # Direct writes through .data/storage can evade version counters: never
        # edit codec storage that way; use validated upgrade_memory_codec().
        tensors = (*self._codec_parameters(), self.codec_to_memory_matrix, self.codec_to_memory_bias)
        versions = tuple((id(t), t._version) for t in tensors)
        if self._fingerprint_cache is not None and self._fingerprint_cache[0] == versions:
            return self._fingerprint_cache[1]
        digest = hashlib.sha256()
        digest.update(self._hash_codec(self.memory_token_embedding, self.memory_encoder, self.memory_state_proj).encode())
        digest.update(self.codec_to_memory_matrix.detach().cpu().contiguous().numpy().tobytes())
        digest.update(self.codec_to_memory_bias.detach().cpu().contiguous().numpy().tobytes())
        result = digest.hexdigest()
        self._fingerprint_cache = (versions, result)
        return result

    @staticmethod
    def _hash_codec(embedding: nn.Module, encoder: nn.Module, projection: nn.Module) -> str:
        digest = hashlib.sha256()
        for module in (embedding, encoder, projection):
            for parameter in module.parameters():
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _legacy_encoder_fingerprint(self) -> str:
        """v0.4.1 fingerprint, for strict backward checkpoint verification only."""
        digest = hashlib.sha256()
        for name, parameter in self.named_parameters():
            if name == 'token_embedding.weight' or name.startswith(('memory_encoder.', 'memory_state_proj.')):
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _encode_anchors(self, texts, tokenizer, *, codec=None):
        if not isinstance(texts, (tuple, list)) or not texts or any(not isinstance(x, str) or not x for x in texts):
            raise ValueError('Nonempty anchor sentences required')
        if codec is None:
            codec = (self.memory_token_embedding, self.memory_encoder, self.memory_state_proj)
        embedding, encoder, projection = codec
        vectors = []
        device = self.memory.states.device
        with torch.no_grad():
            for text in texts:
                ids = tokenizer.encode(text)
                if not 1 <= len(ids) <= self.config.memory_tokens:
                    raise ValueError('Anchor exceeds memory token budget')
                tensor = torch.tensor(ids, dtype=torch.long, device=device)[None]
                body, _ = encoder(embedding(tensor))
                vectors.append(projection(body)[0].detach().to(dtype=torch.float64))
        joined = torch.cat(vectors, dim=0)
        if codec == (self.memory_token_embedding, self.memory_encoder, self.memory_state_proj):
            return joined @ self.codec_to_memory_matrix.double() + self.codec_to_memory_bias.double()
        return joined

    @torch.no_grad()
    def upgrade_memory_codec(self, tokenizer: LayeredTokenizer,
                             embedding: nn.Embedding, encoder: nn.GRU, projection: nn.Linear, *,
                             train_anchors: list[str], validation_anchors: list[str],
                             min_validation_cosine: float = 0.98) -> dict:
        """Explicit validated migration of persistent states into a new learned codec space.

        Uses disjoint anchor sentences to fit and validate an affine state adapter.
        Never changes existing memory / weights on validation failure. For nonlinear
        representation changes the adapter may fail; then retain the old codec.
        """
        if self.memory.count and self.memory.encoder_fingerprint != self._encoder_fingerprint():
            raise RuntimeError('Current memory codec changed outside a validated migration')
        if tokenizer.model_contract() != (self._tokenizer_contract or tokenizer.model_contract()):
            raise ValueError('Tokenizer contract mismatch')
        if not isinstance(min_validation_cosine, (float, int)) or not 0 < min_validation_cosine <= 1:
            raise ValueError('Invalid migration threshold')
        device = self.memory.states.device
        candidates = (copy.deepcopy(embedding).to(device).eval(), copy.deepcopy(encoder).to(device).eval(),
                      copy.deepcopy(projection).to(device).eval())
        if (not isinstance(candidates[0], nn.Embedding) or candidates[0].weight.shape != self.memory_token_embedding.weight.shape
            or not isinstance(candidates[1], nn.GRU) or candidates[1].bidirectional is not True
            or candidates[1].input_size != self.config.d_model or candidates[1].hidden_size != self.config.d_model // 2
            or not isinstance(candidates[2], nn.Linear) or candidates[2].weight.shape != self.memory_state_proj.weight.shape):
            raise ValueError('Codec dimensions incompatible with architecture')
        # Fit candidate raw vectors -> stable neural memory interface. Existing
        # memory bodies and LM reader parameters never need to be rewritten.
        old_train = self._encode_anchors(train_anchors, tokenizer)
        new_train = self._encode_anchors(train_anchors, tokenizer, codec=candidates)
        if not torch.isfinite(new_train).all() or new_train.square().sum() < 1e-12:
            raise ValueError('Candidate codec has unusable representations')
        design = torch.cat([new_train, torch.ones(len(new_train), 1, dtype=torch.float64, device=device)], dim=1)
        fit = torch.linalg.lstsq(design, old_train).solution
        old_validation = self._encode_anchors(validation_anchors, tokenizer)
        new_validation = self._encode_anchors(validation_anchors, tokenizer, codec=candidates)
        predictions = (torch.cat([new_validation, torch.ones(len(new_validation), 1, dtype=torch.float64, device=device)], dim=1) @ fit)
        cosine = torch.nn.functional.cosine_similarity(predictions, old_validation, dim=-1)
        score = float(cosine.mean().item())
        relative = float(((predictions - old_validation).norm(dim=-1) /
                         old_validation.norm(dim=-1).clamp(min=1e-8)).mean().item())
        if (not torch.isfinite(predictions).all() or score < min_validation_cosine
             or float(cosine.min()) < min_validation_cosine or relative > 0.10):
            raise ValueError('Memory codec upgrade failed held-out compatibility validation')
        # No parameter or stored memory is mutated until training/validation
        # of the adapter succeeds. Existing identities, graph and bodies stay fixed.
        count = self.memory.count
        self.memory_token_embedding, self.memory_encoder, self.memory_state_proj = candidates
        self.codec_to_memory_matrix.copy_(fit[:-1].to(self.codec_to_memory_matrix))
        self.codec_to_memory_bias.copy_(fit[-1].to(self.codec_to_memory_bias))
        if count:
            self._freeze_memory_codec()
        self.memory.current_version += 1  # Encoder-interface version, NOT stored-body format.
        self.memory.encoder_fingerprint = self._encoder_fingerprint() if count else None
        record = {'from_version': self.memory.current_version - 1,
                  'to_version': self.memory.current_version,
                  'validation_cosine': score, 'validation_relative_error': relative,
                  'validation_anchor_count': len(new_validation),
                  'min_required_cosine': float(min_validation_cosine)}
        self.memory.migration_history.append(record)
        return {'version': self.memory.current_version,
                'validation_cosine': score, 'validation_relative_error': relative}

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        support_ids: torch.Tensor | None = None,
        support_lengths: torch.Tensor | None = None,
        support_regions: torch.Tensor | None = None,
        return_logits: bool = True,
        loss_chunk_size: int = 128,
    ) -> IntegratedOutput:
        self._check_ids(input_ids)
        if input_ids.size(1) > self.config.block_size:
            raise ValueError("Input context exceeds bounded window")
        if targets is not None and (
            not isinstance(targets, torch.Tensor)
            or targets.shape != input_ids.shape
            or targets.dtype != torch.long
        ):
            raise ValueError("Target IDs must match input, with -100 permitted for ignored targets")
        if targets is not None and (
            torch.any((targets < 0) & (targets != -100))
            or torch.any(targets >= self.config.vocab_size)
        ):
            raise ValueError("Invalid target token IDs")
        if self.memory.count and self.memory.encoder_fingerprint != self._encoder_fingerprint():
            raise RuntimeError("Persistent memory codec changed outside validated migration")
        if (support_ids is None) != (support_lengths is None):
            raise ValueError("Support ids and lengths must be provided together")
        if support_regions is not None and support_ids is None:
            raise ValueError('Support regions require support ids and lengths')
        support = None if support_ids is None else self._encode_support(support_ids, support_lengths)
        if support_regions is not None:
            support = (*support, support_regions)
        if support is not None and support_ids.size(0) != input_ids.size(0):
            raise ValueError("Support batch mismatch")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(lambda h, current=block: current(h, self.memory, support),
                               x, use_reentrant=False)
            else:
                x = block(x, self.memory, support)
        x = self.final_norm(x)
        if not return_logits:
            if targets is None:
                raise ValueError("Training without logits requires target labels")
            if loss_chunk_size < 1:
                raise ValueError("loss_chunk_size must be positive")
            # Checkpoint each output projection+CE to avoid retaining an
            # [8192,50000] logits tensor in the training autograd graph.
            flat = x.reshape(-1, x.size(-1))
            gold = targets.reshape(-1)
            total_valid = (gold != -100).sum().clamp_min(1)
            partial = []
            for start in range(0, flat.size(0), loss_chunk_size):
                hidden = flat[start:start + loss_chunk_size]
                labels = gold[start:start + loss_chunk_size]
                if not bool((labels != -100).any()):
                    continue
                def chunk_loss(h, t):
                    logits = self.lm_head(h)
                    return F.cross_entropy(logits.float(), t,
                                           ignore_index=-100, reduction='sum')
                if self.training and torch.is_grad_enabled():
                    part = checkpoint(chunk_loss, hidden, labels, use_reentrant=False)
                else:
                    part = chunk_loss(hidden, labels)
                partial.append(part)
            loss = sum(partial) / total_valid if partial else flat.sum() * 0
            return IntegratedOutput(None, loss)
        logits = self.lm_head(x)
        loss = (F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1), ignore_index=-100)
                if targets is not None else None)
        return IntegratedOutput(logits, loss)

    @torch.no_grad()
    def remember(self, text: str, tokenizer: LayeredTokenizer, *, source: str, region: int = 0) -> int:
        if not isinstance(tokenizer, (LayeredTokenizer, SentencePieceTokenizer)) or tokenizer.vocab_size != self.config.vocab_size:
            raise ValueError("Tokenizer/model vocabulary mismatch")
        if not isinstance(text, str) or not text:
            raise ValueError("Nonempty text is required")
        contract = tokenizer.model_contract()
        if self._tokenizer_contract is not None and self._tokenizer_contract != contract:
            raise ValueError("Tokenizer changed since neural memory was created")
        ids = tokenizer.encode(text)
        if not 1 <= len(ids) <= self.config.memory_tokens:
            raise ValueError("Memory text exceeds token budget; splitting not implemented")

        device = self.token_embedding.weight.device
        token_ids = torch.tensor(ids, dtype=torch.long, device=device)[None, None, :]
        lengths = torch.tensor([[len(ids)]], dtype=torch.long, device=device)
        was_training = self.training
        self.eval()
        try:
            states, _, _ = self._encode_support(token_ids, lengths)
            memory_id = self.memory.write(
                states[0, 0], source=source, region=region,
                encoder_fingerprint=self._encoder_fingerprint(),
            )
            self._tokenizer_contract = contract
            self._freeze_memory_codec()
            return memory_id
        finally:
            self.train(was_training)

    def _recurrent_step(self, input_ids: torch.Tensor, states):
        """One incremental token step, only available on recurrent models."""
        if self.config.backbone != "recurrent" or input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("Incremental step requires one token and recurrent backbone")
        x = self.token_embedding(input_ids)
        new_states=[]
        for block, state in zip(self.blocks, states):
            x, next_state=block.step(x, self.memory, state)
            new_states.append(next_state)
        return self.lm_head(self.final_norm(x))[:, -1, :], new_states

    def _recurrent_prefill(self, input_ids: torch.Tensor):
        """Whole-prompt parallel scan and initial recurrent states for decoding.

        Avoids serial Python decoding of every prompt token. Inference-only;
        state isn't persisted across calls or changes to external memory.
        """
        if self.config.backbone != 'recurrent':
            raise ValueError('Prefill requires a recurrent backbone')
        self._check_ids(input_ids)
        if input_ids.shape[1] > self.config.block_size:
            raise ValueError('Prefill exceeds configured context')
        if self.memory.count and self.memory.encoder_fingerprint != self._encoder_fingerprint():
            raise RuntimeError('Persistent memory codec changed outside validated migration')
        x = self.token_embedding(input_ids)
        states = []
        for block in self.blocks:
            x, state = block(x, self.memory, return_state=True)
            states.append(state)
        logits = self.lm_head(self.final_norm(x[:, -1]))
        return logits, states

    def recurrent_continue(self, input_ids: torch.Tensor, states, targets: torch.Tensor | None = None):
        """Continue a recurrent stream from *existing* per-layer states.

        The initial states can be updated directly by a learned graph memory
        bridge. Evidence is never concatenated to the token prompt.
        Differentiable: answer loss propagates through states and the bridge.
        No new parameters / no modifications to legacy checkpoint formats.
        """
        if self.config.backbone != 'recurrent':
            raise ValueError('Continuation only supports the recurrent backbone')
        self._check_ids(input_ids)
        if (not isinstance(states, (tuple, list)) or len(states) != len(self.blocks)
                or any(not isinstance(s, torch.Tensor) or
                       s.shape != (input_ids.shape[0], self.config.d_model)
                       for s in states)):
            raise ValueError('Expected one recurrent state [batch,width] per block')
        if targets is not None and (targets.shape != input_ids.shape or
                                     targets.dtype != torch.long or
                                     bool(((targets < 0) & (targets != -100)).any()) or
                                     bool((targets >= self.config.vocab_size).any())):
            raise ValueError('Invalid continuation targets')
        if self.memory.count and self.memory.encoder_fingerprint != self._encoder_fingerprint():
            raise RuntimeError('Persistent memory codec changed outside validated migration')
        x = self.token_embedding(input_ids)
        next_states = []
        for block, initial_state in zip(self.blocks, states):
            x, state = block(x, self.memory, return_state=True, initial_state=initial_state)
            next_states.append(state)
        logits = self.lm_head(self.final_norm(x))
        loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                targets.reshape(-1), ignore_index=-100)
                if targets is not None else None)
        return IntegratedOutput(logits, loss), next_states

    def retrieval_features(self, input_ids: torch.Tensor):
        """Question's recurrent representation used for supervised retrieval gating.

        Use with a frozen LM to train only retrieval_controller. A controller
        cannot infer exact factual correctness from this vector alone.
        """
        if self.config.backbone != 'recurrent':
            raise ValueError('Retrieval controller requires the recurrent backbone')
        self._check_ids(input_ids)
        if input_ids.size(1)>self.config.block_size:
            raise ValueError('Question exceeds configured context')
        x=self.token_embedding(input_ids)
        for block in self.blocks:
            x=block(x,self.memory)
        return self.final_norm(x[:,-1])

    @torch.inference_mode()
    def retrieval_probability(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.config.backbone!='recurrent' or not bool(self.retrieval_controller_trained):
            raise RuntimeError('Retrieval controller must be trained and validated first')
        previous=self.training
        self.eval()
        try:
            return torch.sigmoid(self.retrieval_controller(self.retrieval_features(input_ids)))
        finally:
            self.train(previous)

    @torch.inference_mode()
    def generate_with_graph(self, question: str, tokenizer, archive, graph_bridge, *,
                            ranker=None, **kwargs):
        """v0.6.2 native API: inject graph evidence into *live recurrent states*.

        Unlike `generate_with_archive`, no retrieved source text is placed in
        the language prompt. Bridge weights are opt-in and checkpointed
        separately so existing LM checkpoints keep their exact schema.
        """
        from .graph_bridge import GraphRecurrentEngine
        return GraphRecurrentEngine(self,tokenizer,archive,graph_bridge,
                                    ranker=ranker).generate(question,**kwargs)

    @torch.inference_mode()
    def generate_with_archive(self, question: str, tokenizer, archive, *,
                              top_k: int = 3, max_hops: int = 1,
                              max_evidence_tokens: int = 256, max_new_tokens: int = 40,
                              temperature: float = 0, eos_id: int | None = None,
                              retrieval_mode: str = 'always', controller_threshold: float = 0.5,
                              evidence_ranker=None, candidate_pool: int = 12,
                              prompt_style: str = 'legacy'):
        """Retrieve original text and condition neural generation on that text.

        Retrieval is conservative/always-on for now: there is NO trained
        decision controller yet. Returns (generated_text, evidence_records).
        Evidence is untrusted source data, not system instructions.
        """
        from .context_archive import ContextArchive
        if not isinstance(archive, ContextArchive):
            raise TypeError('Expected ContextArchive')
        if not isinstance(question, str) or not question.strip() or max_evidence_tokens < 0:
            raise ValueError('Question and nonnegative evidence budget required')
        if retrieval_mode not in ('always','never','learned'):
            raise ValueError('retrieval_mode must be always, never, or learned')
        if not 0.0 <= controller_threshold <= 1.0:
            raise ValueError('controller_threshold must be 0..1')
        should_retrieve = retrieval_mode == 'always'
        if retrieval_mode == 'learned':
            question_ids = tokenizer.encode(question)
            if not question_ids:
                raise ValueError('Empty tokenized question')
            q = torch.tensor(question_ids, dtype=torch.long, device=self.token_embedding.weight.device)[None]
            should_retrieve = float(self.retrieval_probability(q)) >= controller_threshold
        if not 1 <= candidate_pool <= 100:
            raise ValueError('candidate_pool must be 1..100')
        if evidence_ranker is not None and self.config.backbone != 'recurrent':
            raise ValueError('Optional evidence reranker is intended for recurrent models')
        hits = archive.retrieve(question, top_k=max(top_k, candidate_pool) if evidence_ranker else top_k,
                                max_hops=max_hops,
                                max_total=max(candidate_pool, top_k) if evidence_ranker else 5) if should_retrieve else []
        if evidence_ranker is not None and hits:
            hits = evidence_ranker.rerank(question, hits, tokenizer, top_k=top_k)
        # Process sources as quotation-only evidence; keep original text
        # unchanged and enforce a strict token budget. The model needs
        # instruction tuning to reliably use this formatting.
        evidence=[]
        remaining=min(max_evidence_tokens, max(0, self.config.block_size - 1))
        for hit in hits:
            chunk=tokenizer.encode(hit['text'])
            if len(chunk) > remaining:
                continue
            evidence.append(hit)
            remaining -= len(chunk)
        prompt = evidence_prompt(question, evidence, style=prompt_style)
        tokens=tokenizer.encode(prompt)
        if len(tokens) > self.config.block_size:
            # Prefer the question over any retrieved evidence; fail rather
            # than silently truncate the user's question.
            while evidence and len(tokens)>self.config.block_size:
                evidence.pop()
                prompt=evidence_prompt(question, evidence, style=prompt_style)
                tokens=tokenizer.encode(prompt)
            if len(tokens)>self.config.block_size:
                raise ValueError('Question too long for context; increase block size')
        tensor=torch.tensor(tokens,device=self.token_embedding.weight.device,dtype=torch.long)[None]
        generated=self.generate(tensor,max_new_tokens=max_new_tokens,
                                temperature=temperature,eos_id=eos_id)
        return tokenizer.decode(generated[0].tolist()[len(tokens):]), evidence

    @torch.inference_mode()
    def generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int = 40,
        temperature: float = 0.8, top_k: int = 40, eos_id: int | None = None,
    ) -> torch.Tensor:
        self._check_ids(input_ids)
        if (
            input_ids.size(0) != 1
            or max_new_tokens < 0
            or not 0 <= temperature
            or top_k < 0
        ):
            raise ValueError("Invalid generation parameters")
        was_training = self.training
        self.eval()
        try:
            ids = input_ids.clone()
            recurrent_states = None
            if self.config.backbone == "recurrent" and max_new_tokens:
                # Only one pass across the prompt; then O(1)-context recurrent
                # steps. State is local to this call and is never shared across
                # conversations (no stale memory/cache contamination).
                if ids.shape[1] > self.config.block_size:
                    raise ValueError('Recurrent prefill input exceeds configured context')
                step_logits, recurrent_states = self._recurrent_prefill(ids)
            for _ in range(max_new_tokens):
                logits = (step_logits if recurrent_states is not None else
                          self(ids[:, -self.config.block_size:]).logits[:, -1, :]).float()
                if temperature == 0:
                    next_id = logits.argmax(-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if 0 < top_k < logits.size(-1):
                        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                ids = torch.cat([ids, next_id], dim=1)
                if eos_id is not None and int(next_id.item()) == eos_id:
                    break
                if recurrent_states is not None:
                    step_logits, recurrent_states = self._recurrent_step(next_id, recurrent_states)
            return ids
        finally:
            self.train(was_training)

    def save(self, path: str | Path, tokenizer: LayeredTokenizer) -> None:
        if not isinstance(tokenizer, (LayeredTokenizer, SentencePieceTokenizer)) or tokenizer.vocab_size != self.config.vocab_size:
            raise ValueError("Tokenizer/model vocabulary mismatch")
        if self._tokenizer_contract is not None and tokenizer.model_contract() != self._tokenizer_contract:
            raise ValueError("Checkpoint tokenizer differs from memory writer tokenizer")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.memory.count and self.memory.encoder_fingerprint != self._encoder_fingerprint():
            raise RuntimeError("Cannot save incompatible memory codec")
        metadata = {
            "schema": self.CHECKPOINT_SCHEMA,
            "config": asdict(self.config),
            "tokenizer_contract": tokenizer.model_contract(),
            "memory": self.memory.snapshot(),
        }
        file_handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=destination.parent)
        os.close(file_handle)
        try:
            torch.save({"meta": metadata, "state_dict": self.state_dict()}, temporary)
            os.replace(temporary, destination)
        finally:
            # A failed write can leave the temporary file briefly locked on
            # Windows. Never hide the original torch.save failure with WinError 32.
            if os.path.exists(temporary):
                for delay in (0, 0.1, 0.3):
                    if delay:
                        time.sleep(delay)
                    try:
                        os.unlink(temporary)
                        break
                    except FileNotFoundError:
                        break
                    except OSError as exc:
                        cleanup_error = exc
                else:
                    warnings.warn(
                        f"Could not remove temporary checkpoint {temporary}: {cleanup_error}",
                        RuntimeWarning, stacklevel=2,
                    )

    @classmethod
    def load(cls, path: str | Path, tokenizer: LayeredTokenizer) -> "IntegratedLM":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint_payload(payload, tokenizer)

    @classmethod
    def _from_checkpoint_payload(cls, payload, tokenizer: LayeredTokenizer) -> "IntegratedLM":
        if not isinstance(payload, dict) or set(payload) != {"meta", "state_dict"}:
            raise ValueError("Invalid checkpoint")
        metadata = payload["meta"]
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") not in (1, cls.CHECKPOINT_SCHEMA)
            or metadata.get("tokenizer_contract") != tokenizer.model_contract()
        ):
            raise ValueError("Checkpoint schema/tokenizer mismatch")
        model = cls(IntegratedConfig(**metadata["config"]))
        if metadata['schema'] == 1:
            # Do not discard old memory bodies or silently adopt changed weights.
            state = payload['state_dict']
            compatible, missing = model.load_state_dict(state, strict=False), {
                'memory_token_embedding.weight', 'memory.versions',
                'codec_to_memory_matrix', 'codec_to_memory_bias'}
            if set(compatible.missing_keys) != missing or compatible.unexpected_keys:
                raise ValueError('Unrecognized legacy checkpoint parameters')
            with torch.no_grad():
                model.memory_token_embedding.weight.copy_(model.token_embedding.weight)
            if int(model.memory.count) and metadata['memory']['encoder_fingerprint'] != model._legacy_encoder_fingerprint():
                raise ValueError('Legacy memory encoder mismatch')
            model.memory.restore_metadata(metadata['memory'], legacy=True)
            model.memory.encoder_fingerprint = model._encoder_fingerprint() if model.memory.count else None
            model.memory.versions[:model.memory.count] = 1
        else:
            model.load_state_dict(payload["state_dict"], strict=True)
            model.memory.restore_metadata(metadata["memory"])
        model._tokenizer_contract = metadata["tokenizer_contract"]
        if model.memory.count and model.memory.encoder_fingerprint != model._encoder_fingerprint():
            raise ValueError("Saved memories incompatible with neural encoder")
        if model.memory.count:
            model._freeze_memory_codec()
        model.eval()
        return model
