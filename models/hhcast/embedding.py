

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        dim: int,
        patch_len: int,
        stride: int,
        num_patches: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        use_position_embedding: bool = True,
    ) -> None:
        super().__init__()
        if patch_len < 1 or stride < 1 or num_patches < 1:
            raise ValueError("patch_len, stride, and num_patches must be positive")
        if dim % n_heads != 0:
            raise ValueError("n_heads must divide d_model")

        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches
        self.patch_projector = nn.Linear(patch_len, dim)
        self.position_embedding = (
            nn.Parameter(torch.empty(1, num_patches, dim))
            if use_position_embedding
            else None
        )
        if self.position_embedding is not None:
            nn.init.trunc_normal_(self.position_embedding, std=0.02)

        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            try:
                self.encoder = nn.TransformerEncoder(
                    layer,
                    num_layers=n_layers,
                    enable_nested_tensor=False,
                )
            except TypeError:
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.encoder = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Channel-first historical input with shape ``[B, C, T]``.

        Returns:
            Encoded patch tokens ``[B, C, Np, d]`` and their mean-pooled
            channel tokens ``[B, C, d]``.
        """

        if x.ndim != 3:
            raise ValueError("x must have shape [B, C, T]")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.shape[2] != self.num_patches:
            raise RuntimeError(
                f"expected {self.num_patches} patches, got {patches.shape[2]}"
            )

        tokens = self.patch_projector(patches)
        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding.unsqueeze(1)

        batch_size, n_channels, n_patches, dim = tokens.shape
        encoded = self.encoder(tokens.reshape(batch_size * n_channels, n_patches, dim))
        encoded = encoded.reshape(batch_size, n_channels, n_patches, dim)
        channel_tokens = encoded.mean(dim=2)
        return encoded, channel_tokens
