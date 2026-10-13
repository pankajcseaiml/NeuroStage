"""
NeuroStage model  =  lightweight CNN (per-epoch)  +  small Transformer (across epochs).

   raw EEG epochs (B, L, 3000)
        |  EpochCNN   : two depthwise-separable 1D-CNN branches (fast + slow waves)
   epoch embeddings (B, L, d)
        |  TemporalEncoder : self-attention over the L epochs (+ position embedding)
   logits (B, L, 5)         -> Wake / N1 / N2 / N3 / REM

Every model in this project exposes the same 3 methods so the rest of the code
does not care which one it is:   embed(x) -> e,   temporal(e) -> (logits, attn),   forward(x).
"""
import math
import torch
import torch.nn as nn
from config import EPOCH_LEN, SEQ_LEN, N_CLASSES


# ---------- building blocks -------------------------------------------------
class DSConv1d(nn.Module):
    """Depthwise-separable conv: one filter per channel, then a 1x1 mix. ~k times fewer weights."""
    def __init__(self, cin, cout, k):
        super().__init__()
        self.depthwise = nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin, bias=False)
        self.pointwise = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class Branch(nn.Module):
    """One CNN branch. Small stem kernel = fine detail (spindles); big kernel = slow waves."""
    def __init__(self, stem_k, stem_stride, ch, pool):
        super().__init__()
        c1, c2, c3 = ch
        self.stem = nn.Sequential(nn.Conv1d(1, c1, stem_k, stem_stride, padding=stem_k // 2, bias=False),
                                  nn.BatchNorm1d(c1), nn.GELU())
        self.body = nn.Sequential(nn.MaxPool1d(pool), DSConv1d(c1, c2, 7),
                                  nn.MaxPool1d(pool), DSConv1d(c2, c3, 7))
        self.out_dim = c3 * 2

    def forward(self, x):                      # x: (N, 1, 3000)
        h = self.body(self.stem(x))
        return torch.cat([h.mean(-1), h.amax(-1)], dim=-1)   # avg + max pooling over time


class EpochCNN(nn.Module):
    def __init__(self, ch, d_model, drop):
        super().__init__()
        self.fast = Branch(stem_k=25, stem_stride=3, ch=ch, pool=4)
        self.slow = Branch(stem_k=200, stem_stride=25, ch=ch, pool=2)
        self.proj = nn.Sequential(nn.Linear(self.fast.out_dim + self.slow.out_dim, d_model),
                                  nn.LayerNorm(d_model), nn.Dropout(drop))

    def forward(self, x):                      # x: (N, 3000)
        x = x.unsqueeze(1)
        return self.proj(torch.cat([self.fast(x), self.slow(x)], dim=-1))


class SelfAttention(nn.Module):
    """Multi-head self-attention written by hand so we can return the attention weights
    (for the clinician heat-map) and so INT8 quantization can see its Linear layers."""
    def __init__(self, d, heads, drop):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv, self.out, self.drop = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Dropout(drop)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            att = att.masked_fill(mask, float("-inf"))
        w = att.softmax(-1)
        y = (self.drop(w) @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(y), w.mean(1)          # weights averaged over heads: (B, L, L)


class TransformerLayer(nn.Module):
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads, drop)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        a, w = self.attn(self.n1(x), mask)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x))), w


# ---------- the full model --------------------------------------------------
DEFAULT_CFG = dict(ch=(16, 32, 64), d_model=64, layers=2, heads=4, ff=128, drop=0.1, causal=False)
TEACHER_CFG = dict(ch=(32, 64, 128), d_model=128, layers=3, heads=4, ff=256, drop=0.1, causal=False)


class NeuroStage(nn.Module):
    def __init__(self, ch=(16, 32, 64), d_model=64, layers=2, heads=4, ff=128, drop=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.cnn = EpochCNN(ch, d_model, drop)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(TransformerLayer(d_model, heads, ff, drop) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_CLASSES)

    def embed(self, x):                         # (N, 3000) -> (N, d)
        return self.cnn(x)

    def temporal(self, e):                      # (B, L, d) -> logits (B, L, 5), attention (B, L, L)
        B, L, _ = e.shape
        x = e + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=e.device), 1) if self.causal else None
        w = None
        for layer in self.layers:
            x, w = layer(x, mask)
        return self.head(self.norm(x)), w

    def forward(self, x, return_attn=False):    # x: (B, L, 3000)
        B, L, T = x.shape
        logits, w = self.temporal(self.embed(x.reshape(B * L, T)).view(B, L, -1))
        return (logits, w) if return_attn else logits
