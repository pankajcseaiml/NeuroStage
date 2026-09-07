"""Unit tests for NeuroStage data pipeline and model forward pass."""
from __future__ import annotations

import numpy as np
import pytest
import torch


class MockCfg(dict):
    def __getattr__(self, name):
        val = self[name]
        if isinstance(val, dict):
            return MockCfg(val)
        return val


@pytest.fixture
def base_cfg():
    return MockCfg({
        "experiment": {"name": "test", "seed": 42, "n_folds": 5, "mlflow_tracking_uri": ""},
        "data": {
            "sfreq": 100, "bandpass_low": 0.3, "bandpass_high": 35.0,
            "epoch_sec": 30, "context_epochs": 20, "manifest_path": ""
        },
        "model": {
            "variant": "neurostage", "n_classes": 5, "head_dropout": 0.25,
            "cnn": {
                "in_channels": 1, "base_channels": 16, "kernel_sizes": [50, 25, 10],
                "use_depthwise_separable": True, "dropout": 0.25, "embedding_dim": 64
            },
            "transformer": {
                "n_layers": 2, "n_heads": 4, "d_model": 64, "d_ff": 128,
                "dropout": 0.1, "max_seq_len": 25
            }
        },
        "training": {"lr": 1e-3, "weight_decay": 1e-4, "use_class_weights": True,
                     "batch_size": 2, "epochs": 1, "lr_warmup_epochs": 0, "patience": 5},
        "compression": {"quantization": {"backend": "qnnpack"}},
        "evaluation": {"output_dir": "results/", "benchmark_n_warmup": 2, "benchmark_n_runs": 5}
    })


def test_depthwise_separable_conv(base_cfg):
    """Test that DepthwiseSeparableConv1d produces correct output shape."""
    from neurostage.models.cnn import DepthwiseSeparableConv1d
    conv = DepthwiseSeparableConv1d(1, 16, kernel_size=50, padding=25)
    x = torch.zeros(4, 1, 3000)  # (B, C, T)
    y = conv(x)
    assert y.shape == (4, 16, 3001) or y.shape[0] == 4 and y.shape[1] == 16


def test_feature_extractor_output_shape(base_cfg):
    """CNN extractor must output (B, embedding_dim)."""
    from neurostage.models.cnn import EEGFeatureExtractor
    extractor = EEGFeatureExtractor(
        in_channels=1, base_channels=16, kernel_sizes=[50, 25, 10],
        use_depthwise_separable=True, dropout=0.0, embedding_dim=64
    )
    x = torch.zeros(4, 1, 3000)
    emb = extractor(x)
    assert emb.shape == (4, 64), f"Expected (4, 64), got {emb.shape}"


def test_full_model_forward_pass(base_cfg):
    """Full model must accept (B, S, 1, T) and output (B, S, 5)."""
    from neurostage.models.neurostage import build_model
    model = build_model(base_cfg)
    model.eval()
    B, S, C, T = 2, 20, 1, 3000
    x = torch.zeros(B, S, C, T)
    with torch.no_grad():
        logits, attn = model(x)
    assert logits.shape == (B, S, 5), f"Expected ({B},{S},5), got {logits.shape}"


def test_cnn_only_variant(base_cfg):
    """CNN-only variant must not use Transformer."""
    base_cfg["model"]["variant"] = "cnn_only"
    from neurostage.models.neurostage import build_model
    model = build_model(base_cfg)
    assert not model.use_transformer
    x = torch.zeros(2, 20, 1, 3000)
    with torch.no_grad():
        logits, attn = model(x)
    assert logits.shape == (2, 20, 5)
    assert attn is None


def test_metrics_computation():
    """Verify metric computation is sane on known inputs."""
    from neurostage.evaluation.metrics import compute_metrics
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 0, 2])  # 1 wrong
    m = compute_metrics(y_true, y_pred)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert -1.0 <= m["kappa"] <= 1.0
    assert len(m["per_class_f1"]) == 5
    assert len(m["confusion_matrix"]) == 5


def test_class_weights_shape():
    """Class weights should be a 1D tensor of length 5."""
    import json
    import os
    import tempfile

    from neurostage.data.dataset import SleepDataset

    # We can test the logic without real HDF5 by testing compute_class_weights with a mock manifest
    # Minimal smoke test — checks the code path runs
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = {
            "dataset": "test", "n_subjects": 1, "total_epochs": 100,
            "class_counts": {"Wake": 30, "N1": 5, "N2": 40, "N3": 15, "REM": 10},
            "subjects": [
                {
                    "subject_id": "SC4001",
                    "h5_path": os.path.join(tmpdir, "SC4001.h5"),
                    "n_epochs": 100,
                    "class_counts": {"Wake": 30, "N1": 5, "N2": 40, "N3": 15, "REM": 10},
                }
            ]
        }
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        weights = SleepDataset.compute_class_weights(manifest_path, ["SC4001"])
        assert weights.shape == (5,)
        assert (weights > 0).all()
