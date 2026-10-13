"""
Modules 3/4/6 at run time: night-long staging, attention maps, saliency, real-time streaming.
"""
import io, time
import numpy as np
import torch
from scipy.signal import welch, resample_poly
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

from config import ARTIFACTS, SEQ_LEN, STAGES, FS, EPOCH_LEN, EPOCH_SEC, N_CLASSES
from preprocess import preprocess, to_epochs
from registry import load_ckpt, n_params


# ---------- reading user files ---------------------------------------------
def load_recording(data: bytes, filename: str, fs: float = FS):
    """.npy (1-D signal) | .npz (signal [+ labels]) | .csv/.txt (first column). Returns (signal, labels|None)."""
    name, labels = filename.lower(), None
    if name.endswith(".npz"):
        z = np.load(io.BytesIO(data), allow_pickle=False)
        sig = z["signal"]; labels = z["labels"] if "labels" in z.files else None
    elif name.endswith(".npy"):
        sig = np.load(io.BytesIO(data), allow_pickle=False)
    elif name.endswith((".csv", ".txt")):
        sig = np.loadtxt(io.StringIO(data.decode()), delimiter=",", ndmin=2)[:, 0]
    else:
        raise ValueError("Use .npy, .npz, .csv or .txt")
    sig = np.asarray(sig, dtype=np.float64).squeeze()
    if sig.ndim != 1 or not np.isfinite(sig).all():
        raise ValueError("Signal must be a 1-D array of finite numbers (single EEG channel)")
    if fs != FS:
        sig = resample_poly(sig, int(FS), int(round(fs)))
    if len(sig) < EPOCH_LEN:
        raise ValueError("Recording is shorter than one 30-second epoch")
    return sig.astype(np.float32), labels


# ---------- sleep statistics (what a clinician reads off a hypnogram) -------
def sleep_stats(stages):
    s = np.asarray(stages); n = len(s); asleep = s != 0
    first = int(np.argmax(asleep)) if asleep.any() else n
    tst = asleep.sum() * EPOCH_SEC / 60
    tib = n * EPOCH_SEC / 60
    after = asleep.nonzero()[0]
    waso = ((s[first:] == 0).sum() * EPOCH_SEC / 60) if asleep.any() else 0
    return dict(time_in_bed_min=round(tib, 1), total_sleep_min=round(tst, 1),
                efficiency_pct=round(100 * tst / tib, 1) if tib else 0,
                sleep_latency_min=round(first * EPOCH_SEC / 60, 1), waso_min=round(float(waso), 1),
                transitions=int((np.diff(s) != 0).sum()),
                stage_pct={st: round(100 * float((s == i).mean()), 1) for i, st in enumerate(STAGES)},
                rem_latency_min=round(float(np.argmax(s == 4) * EPOCH_SEC / 60), 1) if (s == 4).any() else None)


# ---------- the engine --------------------------------------------------------
class Stager:
    def __init__(self, ckpt_path):
        self.model, ck = load_ckpt(ckpt_path)
        self.name, self.quantized = ckpt_path.stem, ck["quantized"]
        self.info = dict(name=self.name, params=n_params(self.model), quantized=self.quantized,
                         causal=bool(getattr(self.model, "causal", False)), trained_on=ck["meta"].get("data"),
                         test_macro_f1=round(ck["meta"].get("test", {}).get("macro_f1", 0), 3))

    @torch.no_grad()
    def _embed_all(self, epochs):
        x = torch.from_numpy(epochs)
        return torch.cat([self.model.embed(x[i:i + 128]) for i in range(0, len(x), 128)])

    @torch.no_grad()
    def predict_night(self, signal, labels=None):
        t0 = time.perf_counter()
        epochs = to_epochs(preprocess(signal)); n = len(epochs)
        emb = self._embed_all(epochs)
        L = min(SEQ_LEN, n); W = n - L + 1
        wins = emb.unfold(0, L, 1).permute(0, 2, 1)             # (W, L, d): every sliding window
        probs_sum, cnt = torch.zeros(n, N_CLASSES), torch.zeros(n, 1)
        attn_w = []
        for b in range(0, W, 256):                               # batches of windows
            logits, attn = self.model.temporal(wins[b:b + 256])
            p = logits.softmax(-1)
            for j in range(L):                                   # window b+w, position j -> epoch b+w+j
                idx = torch.arange(b, b + len(p)) + j
                probs_sum[idx] += p[:, j]; cnt[idx] += 1
            if attn is not None: attn_w.append(attn)
        probs = probs_sum / cnt
        k = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)       # light temporal smoothing
        sm = torch.nn.functional.conv1d(torch.nn.functional.pad(probs.T.unsqueeze(1), (1, 1), mode="replicate"), k)
        probs = (sm.squeeze(1).T); probs = probs / probs.sum(-1, keepdim=True)
        stages = probs.argmax(-1).numpy(); conf = probs.max(-1).values.numpy()

        out = dict(n_epochs=n, stages=stages.tolist(), confidence=np.round(conf, 3).tolist(),
                   probs=np.round(probs.numpy(), 3).tolist(), stats=sleep_stats(stages),
                   uncertain_epochs=int((conf < 0.6).sum()), model=self.info,
                   attention=self._attention_band(attn_w, n, L) if attn_w else None,
                   total_ms=round((time.perf_counter() - t0) * 1000), ms_per_epoch=round((time.perf_counter() - t0) * 1000 / n, 2))
        if labels is not None and len(labels) >= n:
            y = np.asarray(labels[:n])
            out["metrics"] = dict(acc=round(accuracy_score(y, stages), 4), macro_f1=round(f1_score(y, stages, average="macro"), 4),
                                  kappa=round(cohen_kappa_score(y, stages), 4),
                                  confusion=confusion_matrix(y, stages, labels=range(N_CLASSES)).tolist())
            out["true_stages"] = y.tolist()
        return out

    def _attention_band(self, attn_w, n, L):
        """band[i][lag] = how much epoch i's prediction looked at epoch i-lag (lag 0..L-1)."""
        A = torch.cat(attn_w)                                    # (W, L, L)
        band = np.zeros((n, L), dtype=np.float32)
        for i in range(n):
            w = min(max(i - L + 1, 0), len(A) - 1); j = i - w   # window where epoch i is the LAST (or earliest-possible) token
            row = A[w, j, : j + 1].numpy()[::-1]                 # keep only epochs <= i, reorder as lag 0,1,2..
            band[i, : len(row)] = row / (row.sum() + 1e-9)
        return np.round(band, 3).tolist()

    def explain_epoch(self, signal, idx):
        """Gradient x input saliency: which seconds of the EEG pushed the model toward its answer."""
        if self.quantized:
            raise ValueError("Explain needs an FP32 model (INT8 has no gradients)")
        epochs = to_epochs(preprocess(signal)); n = len(epochs)
        if not 0 <= idx < n: raise ValueError("epoch out of range")
        L = min(SEQ_LEN, n); start = min(max(idx - L + 1, 0), n - L); j = idx - start
        x = torch.from_numpy(epochs[start:start + L]).unsqueeze(0).clone().requires_grad_(True)
        with torch.enable_grad():
            logits, attn = self.model(x, return_attn=True)
            probs = logits[0, j].softmax(-1)
            cls = int(probs.argmax()); logits[0, j, cls].backward()
        sal = (x.grad[0, j] * x[0, j]).abs().detach().numpy()
        sal = np.convolve(sal, np.ones(50) / 50, mode="same"); sal = sal / (sal.max() + 1e-9)
        sig = epochs[idx]
        f, p = welch(sig, FS, nperseg=512)
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 15), "beta": (15, 30)}
        bp = {k: float(p[(f >= a) & (f < b)].sum()) for k, (a, b) in bands.items()}
        tot = sum(bp.values()) + 1e-9
        return dict(epoch=idx, stage=STAGES[cls], probs=np.round(probs.detach().numpy(), 3).tolist(),
                    signal=np.round(sig[::2], 3).tolist(), saliency=np.round(sal[::2], 3).tolist(),   # 50 Hz for plotting
                    band_power={k: round(v / tot, 3) for k, v in bp.items()},
                    context_attention=None if attn is None else np.round(attn[0, j, :j + 1].detach().numpy()[::-1], 3).tolist())


class StreamingStager:
    """Real-time mode: feed ONE new epoch at a time. Cost per epoch = 1 CNN pass + 1 tiny Transformer pass."""
    def __init__(self, stager):
        self.s, self.buf = stager, []
        self.mean, self.m2, self.count = 0.0, 0.0, 0     # running statistics (Welford) instead of whole-night z-score

    def push(self, raw_epoch):
        t0 = time.perf_counter()
        from scipy.signal import butter, sosfiltfilt
        x = sosfiltfilt(butter(4, [0.3, 35.0], btype="band", fs=FS, output="sos"), raw_epoch.astype(np.float64))
        for v in x[::10]:                                # update running mean / variance
            self.count += 1; d = v - self.mean; self.mean += d / self.count; self.m2 += d * (v - self.mean)
        std = (self.m2 / max(self.count - 1, 1)) ** 0.5 + 1e-8
        x = np.clip((x - self.mean) / std, -8, 8).astype(np.float32)
        with torch.no_grad():
            self.buf = (self.buf + [self.s.model.embed(torch.from_numpy(x).unsqueeze(0))[0]])[-SEQ_LEN:]
            logits, _ = self.s.model.temporal(torch.stack(self.buf).unsqueeze(0))
            p = logits[0, -1].softmax(-1)
        return dict(stage=int(p.argmax()), confidence=round(float(p.max()), 3),
                    probs=np.round(p.numpy(), 3).tolist(), latency_ms=round((time.perf_counter() - t0) * 1000, 2))


def pick_model():
    for name in ("student", "pruned", "int8", "teacher", "baseline"):
        if (ARTIFACTS / f"{name}.pt").exists():
            return ARTIFACTS / f"{name}.pt"
    raise FileNotFoundError("No trained model in ml/artifacts. Run: python train.py --role student")
