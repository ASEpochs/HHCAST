

from __future__ import annotations

import torch
import torch.nn as nn


class BidirectionalForecastingDecoder(nn.Module):
    """Implement Eqs. (14)--(16) without teacher forcing or a causal mask."""

    def __init__(
        self,
        fusion_dim: int,
        history_len: int,
        pred_len: int,
        output_channels: int,
        decoder_dim: int,
        decoder_layers: int = 1,
        decoder_heads: int = 4,
        dropout: float = 0.0,
        use_bidirectional_refinement: bool = True,
    ) -> None:
        super().__init__()
        if decoder_dim % decoder_heads != 0:
            raise ValueError("decoder_heads must divide decoder_dim")

        self.pred_len = pred_len
        self.output_channels = output_channels
        self.use_bidirectional_refinement = use_bidirectional_refinement

        # Eq. (14): the initial direct forecast uses both the channel-aligned
        # fused representation and that channel's historical input.
        self.initial_decoder = nn.Linear(fusion_dim + history_len, pred_len)

        # Eq. (15): each future position is a token containing all C channels,
        # projected to d_b and augmented with a learnable future-step position.
        self.input_projection = nn.Linear(output_channels, decoder_dim)
        self.future_position_embedding = nn.Parameter(
            torch.empty(1, pred_len, decoder_dim)
        )
        nn.init.trunc_normal_(self.future_position_embedding, std=0.02)

        if decoder_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=max(4 * decoder_dim, 32),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            try:
                self.bidirectional_transformer = nn.TransformerEncoder(
                    layer,
                    num_layers=decoder_layers,
                    enable_nested_tensor=False,
                )
            except TypeError:
                self.bidirectional_transformer = nn.TransformerEncoder(
                    layer, num_layers=decoder_layers
                )
        else:
            self.bidirectional_transformer = nn.Identity()
        self.output_projection = nn.Linear(decoder_dim, output_channels)

        # Eq. (16): sigmoid(0) = 0.5, so both forecasts start equally weighted.
        self.gamma_logit = nn.Parameter(torch.zeros(()))

    def initial_forecast(
        self,
        fused_channels: torch.Tensor,
        historical_window: torch.Tensor,
    ) -> torch.Tensor:
        history_channel_first = historical_window.transpose(1, 2)
        decoder_input = torch.cat([fused_channels, history_channel_first], dim=-1)
        return self.initial_decoder(decoder_input)

    def bidirectional_refine(self, initial_forecast: torch.Tensor) -> torch.Tensor:
        future_tokens = initial_forecast.transpose(1, 2)
        future_tokens = self.input_projection(future_tokens)
        future_tokens = future_tokens + self.future_position_embedding
        refined_tokens = self.bidirectional_transformer(future_tokens)
        return self.output_projection(refined_tokens).transpose(1, 2)

    def forward(
        self,
        fused_channels: torch.Tensor,
        historical_window: torch.Tensor,
    ) -> torch.Tensor:
        initial = self.initial_forecast(fused_channels, historical_window)
        if not self.use_bidirectional_refinement:
            return initial

        refined = self.bidirectional_refine(initial)
        gamma = torch.sigmoid(self.gamma_logit)
        return gamma * initial + (1.0 - gamma) * refined
