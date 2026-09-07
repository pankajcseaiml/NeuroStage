"""
Compact Transformer Encoder for temporal modeling of sleep epochs.

Purpose:
  Sleep is not a series of independent 30-second windows. The brain follows
  physiological transition patterns (e.g. Wake → N1 → N2 → N3 → REM). The Transformer
  models these sequential dependencies by attending to the full context window
  (e.g. the last 20 epochs ≈ 10 minutes of sleep).

Architecture:
  Positional encoding → N_layers × TransformerEncoderLayer (multi-head self-attention
  + FFN + residual + LayerNorm)

  Multi-head self-attention complexity scales quadratically with sequence length.
  At 20 epochs, this is extremely tractable even on CPU.

  We expose attention weights from the final layer so the frontend can visualize
  which epochs the model attended to when predicting a given epoch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).
    Operates on (B, S, D) tensors.
    """

    def __init__(self, d_model: int, max_len: int = 25, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerTemporalEncoder(nn.Module):
    """
    Compact Transformer encoder for EEG epoch sequences.

    Input  : (B, S, d_model)  — batch of epoch embedding sequences
    Output : (B, S, d_model)  — contextualized epoch representations
             attention_weights: (B, S, S) from last layer (for interpretability)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 25,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, S, D) convention
            norm_first=True,    # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self._last_attn_weights: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x : (B, S, d_model)
        returns: (output (B, S, d_model), attention_weights (B, S, S) or None)
        """
        x = self.pos_encoding(x)

        output = x
        attn_weights = None
        for i, layer in enumerate(self.encoder.layers):
            is_last = (i == len(self.encoder.layers) - 1)
            # Pre-LN attention block
            norm_x = layer.norm1(output)
            attn_out, cur_attn = layer.self_attn(
                norm_x, norm_x, norm_x,
                key_padding_mask=src_key_padding_mask,
                need_weights=is_last,
                average_attn_weights=True,
            )
            output = output + layer.dropout1(attn_out)

            # Pre-LN Feed-Forward block
            norm_ff = layer.norm2(output)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(norm_ff))))
            output = output + layer.dropout2(ff_out)

            if is_last:
                attn_weights = cur_attn

        return output, attn_weights
