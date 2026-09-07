"""
NeuroStage full model: 1D-CNN + Transformer + Classification Head.

Three model variants are supported via the 'variant' config key:
  - 'cnn_only'   : CNN feature extractor + classification head (no Transformer)
  - 'neurostage' : Full model (CNN + Transformer + head) — default
  - 'teacher'    : Larger model configuration for knowledge distillation source

The model is designed to be:
  1. Exported to ONNX / TorchScript for deployment
  2. Compatible with torch.quantization for INT8 quantization
  3. Patchable for magnitude-based pruning via torch.nn.utils.prune
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurostage.models.cnn import EEGFeatureExtractor
from neurostage.models.transformer import TransformerTemporalEncoder


class ClassificationHead(nn.Module):
    """Dense + Softmax classification head."""

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuroStage(nn.Module):
    """
    Full NeuroStage model.

    Input  : (B, S, 1, T) — batch of sequences of EEG epochs
             B = batch size, S = context_epochs, 1 = EEG channel, T = samples
    Output :
      logits           : (B, S, n_classes)
      attention_weights: (B, S, S) or None
    """

    def __init__(self, cfg: Any):
        super().__init__()
        cnn_cfg = cfg.model.cnn
        tf_cfg = cfg.model.transformer
        self.variant = cfg.model.variant
        self.n_classes = cfg.model.n_classes

        # CNN Feature Extractor
        self.cnn = EEGFeatureExtractor(
            in_channels=cnn_cfg.in_channels,
            base_channels=cnn_cfg.base_channels,
            kernel_sizes=list(cnn_cfg.kernel_sizes),
            use_depthwise_separable=cnn_cfg.use_depthwise_separable,
            dropout=cnn_cfg.dropout,
            embedding_dim=cnn_cfg.embedding_dim,
        )

        embedding_dim = cnn_cfg.embedding_dim

        # Transformer Temporal Encoder (skipped in cnn_only variant)
        self.use_transformer = self.variant != "cnn_only" and tf_cfg.n_layers > 0
        if self.use_transformer:
            # Ensure d_model matches CNN embedding_dim
            self.transformer = TransformerTemporalEncoder(
                d_model=embedding_dim,
                n_heads=tf_cfg.n_heads,
                n_layers=tf_cfg.n_layers,
                d_ff=tf_cfg.d_ff,
                dropout=tf_cfg.dropout,
                max_seq_len=tf_cfg.max_seq_len,
            )

        # Classification Head
        self.head = ClassificationHead(
            d_model=embedding_dim,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.head_dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x : (B, S, 1, T)
        returns: (logits (B, S, n_classes), attention_weights (B, S, S) or None)
        """
        B, S, C, T = x.shape

        # Apply CNN to each epoch independently: (B*S, 1, T) → (B*S, D)
        x_flat = x.view(B * S, C, T)
        embeddings_flat = self.cnn(x_flat)              # (B*S, D)
        embeddings = embeddings_flat.view(B, S, -1)     # (B, S, D)

        attn_weights = None
        if self.use_transformer:
            embeddings, attn_weights = self.transformer(embeddings)   # (B, S, D)

        # Apply classification head to every position in the sequence
        logits = self.head(embeddings)                  # (B, S, n_classes)

        return logits, attn_weights

    def predict(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Convenience method: run forward pass, return predictions + confidence + attention.
        x : (B, S, 1, T)
        """
        with torch.no_grad():
            logits, attn = self.forward(x)
        probs = F.softmax(logits, dim=-1)           # (B, S, n_classes)
        predicted = probs.argmax(dim=-1)             # (B, S)
        confidence = probs.max(dim=-1).values        # (B, S)
        return {
            "logits": logits,
            "probs": probs,
            "predicted": predicted,
            "confidence": confidence,
            "attention_weights": attn,
        }

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters per module."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_p = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
        tf_p = (
            sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            if self.use_transformer
            else 0
        )
        head_p = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {"total": total, "cnn": cnn_p, "transformer": tf_p, "head": head_p}

    def export_onnx(self, path: str, context_epochs: int = 20, n_samples: int = 3000) -> None:
        """Export model to ONNX for deployment."""
        dummy = torch.zeros(1, context_epochs, 1, n_samples)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["eeg_sequence"],
            output_names=["logits", "attention_weights"],
            dynamic_axes={
                "eeg_sequence": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
        )


def build_model(cfg: Any) -> NeuroStage:
    """Factory function: build model from config."""
    return NeuroStage(cfg)
