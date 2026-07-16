

from __future__ import annotations

import torch
import torch.nn as nn

from .components import ChannelHypergraphEncoder
from .decoder import BidirectionalForecastingDecoder
from .embedding import TemporalEncoder
from .normalization import InstanceNormalizer


class HHCAST(nn.Module):


    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.n_vars = int(configs.c_out)
        self.dim = int(configs.d_model)
        self.d_ff = int(configs.d_ff)
        self.patch_len = int(configs.patch_len)
        self.stride = int(getattr(configs, "stride", None) or self.patch_len)

        if int(configs.enc_in) != self.n_vars:
            raise ValueError("configs.enc_in must equal configs.c_out")
        if self.patch_len > self.seq_len:
            raise ValueError("patch_len cannot exceed seq_len")
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        n_heads = int(configs.n_heads)
        temporal_layers = int(
            getattr(configs, "temporal_layers", None) or configs.e_layers
        )
        decoder_dim = int(getattr(configs, "decoder_dim", None) or self.dim)
        decoder_heads = int(getattr(configs, "decoder_heads", None) or n_heads)
        tau = getattr(configs, "tau", None)
        tau = 0.8 if tau is None else float(tau)

        ablation = getattr(configs, "ablation", "full")
        valid_ablations = {
            "full",
            "no_hhc",
            "no_dcp",
            "global_only",
            "subgroup_only",
            "no_bfd",
        }
        if ablation not in valid_ablations:
            raise ValueError(f"unknown ablation variant: {ablation}")

        use_revin = bool(int(getattr(configs, "use_revin", 0)))
        self.normalizer = InstanceNormalizer(
            int(configs.enc_in),
            affine=bool(int(getattr(configs, "revin_affine", 0))),
            subtract_last=bool(int(getattr(configs, "revin_subtract_last", 0))),
            non_norm=not use_revin,
        )

        self.temporal_encoder = TemporalEncoder(
            dim=self.dim,
            patch_len=self.patch_len,
            stride=self.stride,
            num_patches=self.num_patches,
            n_heads=n_heads,
            n_layers=temporal_layers,
            d_ff=self.d_ff,
            dropout=float(configs.dropout),
            use_position_embedding=bool(int(configs.pos)),
        )

        if ablation == "global_only":
            aggregation_mode = "global_only"
        elif ablation == "subgroup_only":
            aggregation_mode = "subgroup_only"
        else:
            aggregation_mode = "both"

        self.channel_hypergraph_encoder = ChannelHypergraphEncoder(
            dim=self.dim,
            n_heads=n_heads,
            dropout=float(configs.dropout),
            threshold=tau,
            candidate_graph=getattr(configs, "candidate_graph", "threshold"),
            knn_k=int(getattr(configs, "knn_k", 5)),
            use_hhc=ablation != "no_hhc",
            use_dcp=ablation != "no_dcp",
            aggregation_mode=aggregation_mode,
        )

        self.forecasting_decoder = BidirectionalForecastingDecoder(
            fusion_dim=self.dim,
            history_len=self.seq_len,
            pred_len=self.pred_len,
            output_channels=self.n_vars,
            decoder_dim=decoder_dim,
            decoder_layers=int(configs.d_layers),
            decoder_heads=decoder_heads,
            dropout=float(
                getattr(configs, "decoder_dropout", None)
                if getattr(configs, "decoder_dropout", None) is not None
                else configs.dropout
            ),
            use_bidirectional_refinement=ablation != "no_bfd",
        )

    def forward(
        self,
        z_in: torch.Tensor,
        masks=None,
        is_training: bool = False,
        target=None,
        x_mark=None,
        y_mark=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del masks, is_training, target, x_mark, y_mark
        if z_in.ndim != 3:
            raise ValueError("z_in must have shape [B, T, C]")
        if z_in.shape[1:] != (self.seq_len, self.n_vars):
            raise ValueError(
                f"expected input [B, {self.seq_len}, {self.n_vars}], got {tuple(z_in.shape)}"
            )

        # HHC uses the original window and performs its own window-mean removal
        # exactly as Eq. (1). Optional RevIN affects only the neural path.
        original_window = z_in
        network_input = self.normalizer(z_in, "norm")

        _, channel_tokens = self.temporal_encoder(network_input.transpose(1, 2))
        fused_channels, prune_loss, _ = self.channel_hypergraph_encoder(
            original_window,
            channel_tokens,
        )
        forecast_channel_first = self.forecasting_decoder(
            fused_channels,
            network_input,
        )
        forecast = forecast_channel_first.transpose(1, 2)
        forecast = self.normalizer(forecast, "denorm")
        return forecast, prune_loss
