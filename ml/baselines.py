"""
DeepSleepNet-STYLE baseline (re-implemented and reduced by us, NOT the authors' code):
plain (non-separable) wide CNN + bidirectional LSTM. We train & measure it exactly like
NeuroStage so the accuracy-vs-efficiency comparison is fair and reproducible.
"""
import torch
import torch.nn as nn
from config import N_CLASSES


def _branch(k, stride, pool):
    c = lambda i, o, k_, s=1: [nn.Conv1d(i, o, k_, s, padding=k_ // 2, bias=False), nn.BatchNorm1d(o), nn.ReLU()]
    return nn.Sequential(*c(1, 64, k, stride), nn.MaxPool1d(pool), *c(64, 128, 8), *c(128, 128, 8),
                         *c(128, 128, 8), nn.MaxPool1d(2), nn.AdaptiveAvgPool1d(4), nn.Flatten())


class DeepSleepNetLite(nn.Module):
    def __init__(self, hidden=128, drop=0.3):
        super().__init__()
        self.small, self.large = _branch(50, 6, 8), _branch(400, 50, 4)
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(128 * 4 * 2, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, N_CLASSES)

    def embed(self, x):
        x = x.unsqueeze(1)
        return self.drop(torch.cat([self.small(x), self.large(x)], dim=-1))

    def temporal(self, e):
        h, _ = self.lstm(e)
        return self.head(self.drop(h)), None   # no attention weights in an LSTM

    def forward(self, x, return_attn=False):
        B, L, T = x.shape
        logits, w = self.temporal(self.embed(x.reshape(B * L, T)).view(B, L, -1))
        return (logits, w) if return_attn else logits
