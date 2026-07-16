

from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

import torch
import torch.nn as nn


CandidateGraphMode = Literal["threshold", "topk"]
AggregationMode = Literal["both", "global_only", "subgroup_only"]


def connected_components(adjacency: torch.Tensor) -> torch.Tensor:


    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must have shape [B, C, C]")

    batch_size, n_channels, _ = adjacency.shape
    if n_channels == 0:
        raise ValueError("at least one channel is required")

    with torch.no_grad():
        labels = torch.arange(n_channels, device=adjacency.device).expand(batch_size, -1).clone()
        sentinel = torch.full(
            (1,), n_channels, device=adjacency.device, dtype=labels.dtype
        )

        # Min-label propagation computes exactly the connected components.
        # Similarity graphs normally converge in only a few iterations; the
        # channel count is a strict upper bound (a path graph is the worst case).
        for _ in range(n_channels):
            neighbour_labels = torch.where(
                adjacency,
                labels[:, None, :],
                sentinel,
            ).amin(dim=-1)
            new_labels = torch.minimum(labels, neighbour_labels)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels

        compact_labels = []
        for sample_labels in labels:
            _, inverse = torch.unique(sample_labels, sorted=True, return_inverse=True)
            compact_labels.append(inverse)
        return torch.stack(compact_labels, dim=0)


class HigherOrderHypergraphConstruction(nn.Module):
    """Construct the window-specific partition hypergraph in Eqs. (1)--(5)."""

    def __init__(
        self,
        threshold: float = 0.8,
        mode: CandidateGraphMode = "threshold",
        knn_k: int = 5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("the cosine-similarity threshold must be in [-1, 1]")
        if mode not in {"threshold", "topk"}:
            raise ValueError("candidate graph mode must be 'threshold' or 'topk'")
        if knn_k < 1:
            raise ValueError("knn_k must be positive")

        self.threshold = float(threshold)
        self.mode = mode
        self.knn_k = int(knn_k)
        self.eps = float(eps)

    def frequency_domain_similarity(self, x: torch.Tensor) -> torch.Tensor:


        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C]")
        if x.shape[1] < 2:
            raise ValueError("frequency similarity needs at least two time steps")

        channel_first = x.transpose(1, 2)
        centred = channel_first - channel_first.mean(dim=-1, keepdim=True)
        # rFFT contains the zero-frequency coefficient at index 0. Removing it
        # leaves floor(T / 2) coefficients, exactly as specified in Eq. (1).
        amplitude = torch.fft.rfft(centred, dim=-1).abs()[..., 1:]
        norms = torch.linalg.vector_norm(amplitude, ord=2, dim=-1, keepdim=True)
        normalised = amplitude / norms.clamp_min(self.eps)
        similarity = normalised @ normalised.transpose(-1, -2)
        return similarity.clamp(min=-1.0, max=1.0)

    def adjacency_from_similarity(self, similarity: torch.Tensor) -> torch.Tensor:
        """Build either the main threshold graph or the Section 4.5 Top-Knn graph."""

        batch_size, n_channels, _ = similarity.shape
        eye = torch.eye(n_channels, device=similarity.device, dtype=torch.bool)

        if self.mode == "threshold":
            adjacency = similarity > self.threshold
            adjacency = adjacency | adjacency.transpose(-1, -2)
        else:
            if n_channels == 1:
                return eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
            k = min(self.knn_k, n_channels - 1)
            scores = similarity.masked_fill(eye.unsqueeze(0), float("-inf"))
            neighbour_indices = scores.topk(k=k, dim=-1).indices
            directed = torch.zeros_like(similarity, dtype=torch.bool)
            directed.scatter_(-1, neighbour_indices, True)
            adjacency = directed | directed.transpose(-1, -2)

        return adjacency | eye.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        similarity = self.frequency_domain_similarity(x)
        adjacency = self.adjacency_from_similarity(similarity)
        return connected_components(adjacency), similarity


class DifferentiableContextualPruning(nn.Module):
    """Assign one continuous context weight to each candidate hyperedge."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        channel_tokens: torch.Tensor,
        component_labels: torch.Tensor,
        enabled: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:


        weighted_channels: list[torch.Tensor] = []
        weighted_descriptors: list[torch.Tensor] = []
        per_window_penalties: list[torch.Tensor] = []

        for sample_tokens, sample_labels in zip(channel_tokens, component_labels):
            n_edges = int(sample_labels.max().item()) + 1
            descriptors_tensor = sample_tokens.new_zeros(n_edges, sample_tokens.shape[-1])
            descriptors_tensor.index_add_(0, sample_labels, sample_tokens)
            counts = torch.bincount(sample_labels, minlength=n_edges).to(sample_tokens.dtype)
            descriptors_tensor = descriptors_tensor / counts.clamp_min(1).unsqueeze(-1)

            if enabled:
                relevance = torch.sigmoid(self.scorer(descriptors_tensor)).squeeze(-1)
                penalty = relevance.mean()
            else:
                relevance = descriptors_tensor.new_ones(n_edges)
                penalty = descriptors_tensor.new_zeros(())

            channel_relevance = relevance[sample_labels]
            weighted_channels.append(sample_tokens * channel_relevance.unsqueeze(-1))
            weighted_descriptors.append(descriptors_tensor * relevance.unsqueeze(-1))
            per_window_penalties.append(penalty)

        prune_loss = torch.stack(per_window_penalties).mean()
        return weighted_channels, weighted_descriptors, prune_loss


class HierarchicalSparseAggregation(nn.Module):


    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        mode: AggregationMode = "both",
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        if mode not in {"both", "global_only", "subgroup_only"}:
            raise ValueError("invalid aggregation mode")

        self.dim = dim
        self.n_heads = n_heads
        self.mode = mode
        self.global_attention = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.local_attention = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.global_query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.global_query, std=dim ** -0.5)
        self.fusion = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _global_aggregate(self, descriptors: list[torch.Tensor]) -> torch.Tensor:
        max_edges = max(item.shape[0] for item in descriptors)
        padded_samples = []
        valid_masks = []
        for item in descriptors:
            pad_size = max_edges - item.shape[0]
            if pad_size:
                item = torch.cat([item, item.new_zeros(pad_size, self.dim)], dim=0)
            padded_samples.append(item)
            valid_masks.append(
                torch.arange(max_edges, device=item.device) < max_edges - pad_size
            )

        padded = torch.stack(padded_samples, dim=0)
        valid = torch.stack(valid_masks, dim=0)
        attended, _ = self.global_attention(
            padded,
            padded,
            padded,
            key_padding_mask=~valid,
            need_weights=False,
        )
        logits = (attended * self.global_query).sum(dim=-1) / math.sqrt(self.dim)
        logits = logits.masked_fill(~valid, float("-inf"))
        pooling_weights = torch.softmax(logits, dim=-1)
        return torch.sum(attended * pooling_weights.unsqueeze(-1), dim=1)

    def _local_aggregate(
        self,
        channels: list[torch.Tensor],
        component_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply shared local MHA without padding unrelated hyperedges together."""

        buckets: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
        for batch_index, labels in enumerate(component_labels):
            n_edges = int(labels.max().item()) + 1
            counts = torch.bincount(labels, minlength=n_edges)
            ordered_indices = torch.argsort(labels, stable=True)
            for indices in torch.split(ordered_indices, counts.tolist()):
                buckets[int(indices.numel())].append((batch_index, indices))

        outputs = [torch.zeros_like(item) for item in channels]
        for _, entries in buckets.items():
            grouped_tokens = torch.stack(
                [channels[batch_index][indices] for batch_index, indices in entries],
                dim=0,
            )
            attended, _ = self.local_attention(
                grouped_tokens,
                grouped_tokens,
                grouped_tokens,
                need_weights=False,
            )
            for item_index, (batch_index, indices) in enumerate(entries):
                outputs[batch_index][indices] = attended[item_index]

        return torch.stack(outputs, dim=0)

    def forward(
        self,
        weighted_channels: list[torch.Tensor],
        weighted_descriptors: list[torch.Tensor],
        component_labels: torch.Tensor,
    ) -> torch.Tensor:
        local = self._local_aggregate(weighted_channels, component_labels)
        global_summary = self._global_aggregate(weighted_descriptors)
        global_broadcast = global_summary.unsqueeze(1).expand(-1, local.shape[1], -1)

        if self.mode == "global_only":
            local = torch.zeros_like(local)
        elif self.mode == "subgroup_only":
            global_broadcast = torch.zeros_like(global_broadcast)

        fused = self.fusion(torch.cat([local, global_broadcast], dim=-1))
        return self.dropout(fused)


class ChannelHypergraphEncoder(nn.Module):


    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float,
        threshold: float,
        candidate_graph: CandidateGraphMode,
        knn_k: int,
        use_hhc: bool = True,
        use_dcp: bool = True,
        aggregation_mode: AggregationMode = "both",
    ) -> None:
        super().__init__()
        self.use_hhc = use_hhc
        self.use_dcp = use_dcp
        self.construction = HigherOrderHypergraphConstruction(
            threshold=threshold,
            mode=candidate_graph,
            knn_k=knn_k,
        )
        self.pruning = DifferentiableContextualPruning(dim)
        self.aggregation = HierarchicalSparseAggregation(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            mode=aggregation_mode,
        )

    def forward(
        self,
        historical_window: torch.Tensor,
        channel_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_hhc:
            component_labels, similarity = self.construction(historical_window)
        else:
            batch_size, _, n_channels = historical_window.shape
            component_labels = torch.zeros(
                batch_size,
                n_channels,
                device=historical_window.device,
                dtype=torch.long,
            )
            similarity = self.construction.frequency_domain_similarity(historical_window)

        weighted_channels, weighted_descriptors, prune_loss = self.pruning(
            channel_tokens,
            component_labels,
            enabled=self.use_dcp,
        )
        fused = self.aggregation(
            weighted_channels,
            weighted_descriptors,
            component_labels,
        )
        return fused, prune_loss, similarity


# Backward-compatible names for external imports from earlier revisions.
HighOrderHypergraphConstruction = HigherOrderHypergraphConstruction
DifferentiableCausalPruning = DifferentiableContextualPruning
