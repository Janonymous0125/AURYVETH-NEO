"""Persistent neural memory storage, region routing and bounded neural reads.

Graph edges are ordinary association metadata, never neural processing weights.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class NeuralMemoryBank(nn.Module):
    """Bounded neural memory store with region routing and typed graph links.

    Neural bodies are persistent model buffers and never parameters. Retrieval
    remains an O(memory count) reference algorithm, not a scalable index.
    """

    def __init__(self, config: IntegratedConfig):
        super().__init__()
        self.cfg = config
        self.register_buffer("states", torch.zeros(config.memory_slots, config.memory_tokens, config.d_model))
        self.register_buffer("keys", torch.zeros(config.memory_slots, config.d_model))
        self.register_buffer("lengths", torch.zeros(config.memory_slots, dtype=torch.long))
        self.register_buffer("region_ids", torch.zeros(config.memory_slots, dtype=torch.long))
        self.register_buffer("status_codes", torch.zeros(config.memory_slots, dtype=torch.long))
        self.register_buffer("versions", torch.zeros(config.memory_slots, dtype=torch.long))
        self.register_buffer("count_value", torch.zeros((), dtype=torch.long))

        self.provenance: list[dict] = []
        self.edges: list[dict] = []
        self.encoder_fingerprint: str | None = None
        self.current_version = 1
        self.migration_history: list[dict] = []
        self._suspended = False
        self._edge_revision = 0
        self._adjacency_cache = None

    @property
    def count(self) -> int:
        return int(self.count_value.item())

    @contextmanager
    def suspended(self):
        """Temporarily omit persistent memories for an ablation test."""
        previous = self._suspended
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = previous

    @torch.no_grad()
    def write(self, states: torch.Tensor, *, source: str, region: int, encoder_fingerprint: str) -> int:
        if not isinstance(source, str) or not source.strip() or len(source) > 256:
            raise ValueError("Memory source provenance required")
        if type(region) is not int or not 0 <= region < self.cfg.regions:
            raise ValueError("Invalid region id")
        if self.count >= self.cfg.memory_slots:
            raise ValueError("Memory capacity exceeded")
        if (
            not isinstance(states, torch.Tensor)
            or states.ndim != 2
            or states.size(-1) != self.cfg.d_model
            or not 1 <= states.size(0) <= self.cfg.memory_tokens
            or not torch.isfinite(states).all()
        ):
            raise ValueError("Memory must contain finite, bounded neural states")
        if torch.linalg.vector_norm(states.mean(0)) <= 1e-8:
            raise ValueError("Neural memory has no usable retrieval key")
        if self.encoder_fingerprint is not None and self.encoder_fingerprint != encoder_fingerprint:
            raise RuntimeError("Memory encoder changed; migration required before new memory writes")

        index = self.count
        body = states.detach().to(self.states)
        self.states[index].zero_()
        self.states[index, :len(body)].copy_(body)
        self.keys[index].copy_(F.normalize(body.mean(dim=0), dim=0, eps=1e-8))
        self.lengths[index] = len(body)
        self.region_ids[index] = region
        self.status_codes[index] = 1  # Provisional; 2 means consolidated.
        self.versions[index] = 1  # Canonical memory-body format stays stable across encoder upgrades.
        self.count_value += 1
        self.encoder_fingerprint = encoder_fingerprint
        self.provenance.append({"id": index + 1, "region": region, "source": source, "status": "provisional"})
        return index + 1

    def link(self, source_id: int, target_id: int, *, relation: str) -> None:
        if (
            type(source_id) is not int
            or type(target_id) is not int
            or source_id == target_id
            or not 1 <= source_id <= self.count
            or not 1 <= target_id <= self.count
        ):
            raise ValueError("Associations require two existing, distinct memory identities")
        if not isinstance(relation, str) or not 1 <= len(relation.strip()) <= 64:
            raise ValueError("Typed memory association is required")
        edge = {"source_id": source_id, "target_id": target_id, "relation": relation}
        if edge not in self.edges:
            if len(self.edges) >= 4096:
                raise ValueError("Memory graph edge capacity exceeded")
            self.edges.append(edge)
            self._edge_revision += 1
            self._adjacency_cache = None

    def consolidate(self, memory_id: int, *, reviewer: str, evidence: str) -> None:
        if type(memory_id) is not int or not 1 <= memory_id <= self.count:
            raise ValueError("Invalid memory")
        if not reviewer or not evidence:
            raise ValueError("Consolidation requires explicit validation evidence")
        self.status_codes[memory_id - 1] = 2
        previous = self.provenance[memory_id - 1]
        self.provenance[memory_id - 1] = {
            **previous,
            "status": "consolidated",
            "reviewer": reviewer,
            "evidence": evidence,
        }

    def snapshot(self) -> dict:
        if len(self.provenance) != self.count:
            raise RuntimeError("Memory provenance/count mismatch")
        return {
            "provenance": self.provenance,
            "encoder_fingerprint": self.encoder_fingerprint,
            "edges": self.edges,
            "current_version": self.current_version,
            "migration_history": self.migration_history,
        }

    def restore_metadata(self, metadata: dict, *, legacy: bool = False) -> None:
        if (
            not isinstance(metadata, dict)
            or set(metadata) != ({"provenance", "encoder_fingerprint", "edges"} if legacy else
                                  {"provenance", "encoder_fingerprint", "edges", "current_version", "migration_history"})
            or len(metadata["provenance"]) != self.count
        ):
            raise ValueError("Invalid memory metadata")
        for index, item in enumerate(metadata["provenance"]):
            if (
                not isinstance(item, dict)
                or item.get("id") != index + 1
                or item.get("region") != int(self.region_ids[index])
                or not isinstance(item.get("source"), str)
                or item.get("status") not in {"provisional", "consolidated"}
            ):
                raise ValueError("Invalid memory provenance")
            expected = 1 if item["status"] == "provisional" else 2
            if int(self.status_codes[index]) != expected:
                raise ValueError("Memory provenance/status mismatch")

        version = 1 if legacy else metadata['current_version']
        history = [] if legacy else metadata['migration_history']
        if type(version) is not int or version < 1 or not isinstance(history, list):
            raise ValueError('Invalid memory representation version or history')
        if not legacy and (any(int(v) != 1 for v in self.versions[:self.count])
                           or len(history) != version - 1):
            raise ValueError('Memory representation versions inconsistent')
        self.current_version = version
        self.migration_history = history
        self.provenance = metadata["provenance"]
        self.encoder_fingerprint = metadata["encoder_fingerprint"]
        self.edges = []
        if not isinstance(metadata["edges"], list) or len(metadata["edges"]) > 4096:
            raise ValueError("Invalid memory graph")
        for edge in metadata["edges"]:
            if not isinstance(edge, dict) or set(edge) != {"source_id", "target_id", "relation"}:
                raise ValueError("Invalid graph edge")
            self.link(edge["source_id"], edge["target_id"], relation=edge["relation"])
        if len(self.edges) != len(metadata["edges"]):
            raise ValueError("Duplicate graph edge")

    def _graph_adjacency(self, device: torch.device) -> torch.Tensor:
        """Cached adjacency indexed by memory ID; rebuilt on graph edits."""
        cache_key=(self.count,self._edge_revision,str(device))
        if self._adjacency_cache is not None and self._adjacency_cache[0]==cache_key:
            return self._adjacency_cache[1]
        adjacency=torch.zeros((self.count,self.count),dtype=torch.bool,device=device)
        if self.edges:
            src=torch.tensor([e['source_id']-1 for e in self.edges],dtype=torch.long,device=device)
            dst=torch.tensor([e['target_id']-1 for e in self.edges],dtype=torch.long,device=device)
            adjacency[src,dst]=True
            adjacency[dst,src]=True
        self._adjacency_cache=(cache_key,adjacency)
        return adjacency

    def select(self, queries: torch.Tensor, *, support=None):
        """Select [batch, time, memories, segments, width] without future-token leakage.

        ``support`` contains (states, lengths, keys) for differentiable
        training. Otherwise all candidates come from the persistent bank.
        """
        batch, time, width = queries.shape
        if support is None:
            if self._suspended or self.count == 0:
                return None, None
            count = self.count
            values = self.states[:count].unsqueeze(0).expand(batch, -1, -1, -1)
            lengths = self.lengths[:count].unsqueeze(0).expand(batch, -1)
            keys = self.keys[:count].unsqueeze(0).expand(batch, -1, -1)
            region_ids = self.region_ids[:count].unsqueeze(0).expand(batch, -1)
        else:
            if len(support) not in (3, 4):
                raise ValueError('Invalid differentiable support memory')
            values, lengths, keys = support[:3]
            if (
                values.ndim != 4
                or values.size(0) != batch
                or values.size(-1) != width
                or lengths.shape != values.shape[:2]
                or keys.shape != (*values.shape[:2], width)
            ):
                raise ValueError("Invalid differentiable support memory")
            region_ids = torch.zeros_like(lengths) if len(support) == 3 else support[3]
            if (not isinstance(region_ids, torch.Tensor) or region_ids.shape != lengths.shape
                    or region_ids.dtype != torch.long or region_ids.device != lengths.device
                    or bool((region_ids < 0).any()) or bool((region_ids >= self.cfg.regions).any())):
                raise ValueError('Invalid support regions')

        candidates = values.size(1)
        if not candidates:
            return None, None
        query_keys = F.normalize(queries, dim=-1, eps=1e-8)
        scores = torch.einsum("btd,bnd->btn", query_keys, F.normalize(keys, dim=-1, eps=1e-8))

        # Query-position-specific routing through region prototypes.
        active_region_count = min(self.cfg.regions, candidates) if support is None or len(support) == 4 else 1
        prototypes = torch.zeros((batch, self.cfg.regions, width), device=keys.device, dtype=keys.dtype)
        region_counts = torch.zeros((batch, self.cfg.regions, 1), device=keys.device, dtype=keys.dtype)
        prototypes.scatter_add_(1, region_ids[:, :, None].expand(-1, -1, width), keys)
        region_counts.scatter_add_(
            1, region_ids[:, :, None], torch.ones_like(lengths, dtype=keys.dtype)[:, :, None]
        )
        available = region_counts[..., 0] > 0
        prototypes = F.normalize(prototypes / region_counts.clamp(min=1), dim=-1, eps=1e-8)
        region_scores = torch.einsum("btd,brd->btr", query_keys, prototypes)
        region_scores = region_scores.masked_fill(~available[:, None, :], -1e8)
        selected_regions = region_scores.topk(min(self.cfg.active_regions, active_region_count), dim=-1).indices
        permitted = (region_ids[:, None, :, None] == selected_regions[:, :, None, :]).any(-1)
        scores = scores.masked_fill(~permitted, -1e8)

        selection_size = min(self.cfg.active_memories, candidates)
        selected_indices = scores.topk(selection_size, dim=-1).indices
        batch_indices = torch.arange(batch, device=queries.device)[:, None, None].expand_as(selected_indices)
        graph_adj = None
        if support is None and self.edges and selection_size > 1:
            # Vectorized over batch/tokens. Build the adjacency once per call,
            # rather than looping over all tokens and all graph edges. The
            # bounded bank remains intentionally small (<=4096 entries).
            graph_adj = self._graph_adjacency(queries.device)
            anchor=selected_indices[...,0]
            # Lowest-ID neighbor is a stable fallback in the final slot.
            neighbors=graph_adj[anchor]
            preferred=neighbors.int().argmax(-1)
            has_neighbor=neighbors.any(-1)
            already=(selected_indices == preferred[...,None]).any(-1)
            replace=has_neighbor & ~already
            selected_indices=torch.cat(
                [selected_indices[...,:-1],
                 torch.where(replace,preferred,selected_indices[...,-1])[...,None]],dim=-1)

        selected_values = values[batch_indices, selected_indices]
        selected_lengths = lengths[batch_indices, selected_indices]
        valid = permitted.gather(-1, selected_indices)
        if graph_adj is not None:
            neighbor_mask=graph_adj[selected_indices[...,0,None],selected_indices]
            valid = valid | neighbor_mask

        segment_indices = torch.arange(values.size(2), device=queries.device)
        segment_valid = segment_indices[None, None, None, :] < selected_lengths[..., None]
        return selected_values, segment_valid & valid[..., None]


class NeuralMemoryRead(nn.Module):
    """Projects activated neural memory into short-lived attention-style keys/values."""

    def __init__(self, config: IntegratedConfig):
        super().__init__()
        width = config.d_model
        self.to_query = nn.Linear(width, width, bias=False)
        self.to_key = nn.Linear(width, width, bias=False)
        self.to_value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, width)
        self.register_buffer("calls", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, memory: NeuralMemoryBank, support=None) -> torch.Tensor:
        query = self.to_query(x)
        states, mask = memory.select(query, support=support)
        if states is None or not torch.any(mask):
            return torch.zeros_like(x)
        if not self.training:
            self.calls += 1

        batch, time, memories, segments, width = states.shape
        keys = self.to_key(states).reshape(batch, time, memories * segments, width)
        values = self.to_value(states).reshape(batch, time, memories * segments, width)
        scores = torch.einsum("btd,btnd->btn", query, keys) / math.sqrt(width)
        flattened_mask = mask.reshape(batch, time, memories * segments)
        scores = scores.masked_fill(~flattened_mask, -1e8)
        attention = F.softmax(scores, dim=-1) * flattened_mask
        attention = attention / (attention.sum(-1, keepdim=True) + 1e-9)
        read = torch.einsum("btn,btnd->btd", attention, values)
        return torch.sigmoid(self.gate(x)) * self.output(read)
