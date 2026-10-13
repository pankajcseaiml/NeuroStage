"""
Train a model.      python train.py --role student --data synthetic
  --role baseline : DeepSleepNet-style CNN+BiLSTM
  --role teacher  : big NeuroStage (used for knowledge distillation)
  --role student  : small NeuroStage (add --teacher artifacts/teacher.pt to distil)
Splits are SUBJECT-INDEPENDENT: a person is either in train or in test, never both.
"""
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from config import ARTIFACTS, SEQ_LEN, STAGES, N_CLASSES
from preprocess import preprocess, to_epochs
from registry import build, save_ckpt, load_ckpt, n_params
from model import DEFAULT_CFG, TEACHER_CFG


# ---------- data ------------------------------------------------------------
class WindowDataset(Dataset):
    """Every item = SEQ_LEN consecutive epochs from one subject + their labels."""
    def __init__(self, subjects, stride):
        self.x, self.y, self.idx = [], [], []
        for s, (sig, lab) in enumerate(subjects):
            ep = to_epochs(preprocess(sig))
            lab = lab[: len(ep)]
            self.x.append(torch.from_numpy(ep)); self.y.append(torch.from_numpy(lab))
            self.idx += [(s, i) for i in range(0, len(ep) - SEQ_LEN + 1, stride)]

    def __len__(self): return len(self.idx)

    def __getitem__(self, k):
        s, i = self.idx[k]
        return self.x[s][i:i + SEQ_LEN], self.y[s][i:i + SEQ_LEN]


def get_cohort(args):
    if args.data == "sleepedf":
        from data_sleepedf import load_sleepedf
        return load_sleepedf(args.subjects)
    from synthetic import generate_cohort
    return generate_cohort(args.subjects, args.hours, seed=args.seed)


def split(cohort, test_frac=0.2, val_frac=0.2, fold=None, folds=None):
    n = len(cohort)
    if folds:                                           # k-fold over SUBJECTS
        test_ids = list(range(n))[fold::folds]
    else:
        test_ids = list(range(n - max(1, int(n * test_frac)), n))
    rest = [i for i in range(n) if i not in test_ids]
    nv = max(1, int(len(rest) * val_frac))
    val_ids, train_ids = rest[-nv:], rest[:-nv]
    pick = lambda ids: [cohort[i] for i in ids]
    return pick(train_ids), pick(val_ids), pick(test_ids)


def class_weights(subjects):
    counts = np.bincount(np.concatenate([l for _, l in subjects]), minlength=N_CLASSES) + 1
    w = (counts.sum() / (N_CLASSES * counts)) ** 0.5     # sqrt-inverse freq: helps N1 w/o exploding
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ---------- evaluation ------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x).argmax(-1).reshape(-1)); ys.append(y.reshape(-1))
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return dict(acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
                kappa=cohen_kappa_score(y, p),
                per_class_f1=dict(zip(STAGES, f1_score(y, p, average=None, labels=range(N_CLASSES)).round(3).tolist())))


# ---------- training --------------------------------------------------------
def fit(model, train_dl, val_dl, weights, epochs, lr=2e-3, teacher=None, kd_T=2.0, kd_alpha=0.5, log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))
    best, best_state = -1, None
    for ep in range(1, epochs + 1):
        model.train(); t0, tot = time.time(), 0.0
        for x, y in train_dl:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), y.reshape(-1), weight=weights)
            if teacher is not None:                      # knowledge distillation
                with torch.no_grad():
                    t_logits = teacher(x)
                kd = F.kl_div(F.log_softmax(logits / kd_T, -1), F.softmax(t_logits / kd_T, -1),
                              reduction="batchmean") * kd_T ** 2 / logits.shape[1]
                loss = kd_alpha * kd + (1 - kd_alpha) * loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        m = evaluate(model, val_dl)
        log(f"  epoch {ep:02d}/{epochs}  loss {tot/len(train_dl):.3f}  val acc {m['acc']:.3f}  "
            f"macroF1 {m['macro_f1']:.3f}  ({time.time()-t0:.0f}s)")
        if m["macro_f1"] > best:
            best, best_state = m["macro_f1"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def loaders(train, val, test, bs):
    return (DataLoader(WindowDataset(train, SEQ_LEN // 2), bs, shuffle=True, drop_last=True),
            DataLoader(WindowDataset(val, SEQ_LEN), bs),
            DataLoader(WindowDataset(test, SEQ_LEN), bs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["baseline", "teacher", "student"], default="student")
    ap.add_argument("--data", choices=["synthetic", "sleepedf"], default="synthetic")
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--causal", action="store_true", help="strict real-time: no looking at future epochs")
    ap.add_argument("--teacher", default=None, help="checkpoint to distil from")
    ap.add_argument("--folds", type=int, default=0, help="k-fold cross-validation over subjects")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    cohort = get_cohort(a)
    cfg = {"teacher": TEACHER_CFG, "student": DEFAULT_CFG, "baseline": None}[a.role]
    cfg = dict(cfg, causal=a.causal) if cfg else None
    kind = "baseline" if a.role == "baseline" else "neurostage"
    teacher = load_ckpt(a.teacher)[0] if a.teacher else None

    fold_list = range(a.folds) if a.folds else [None]
    results = []
    for f in fold_list:
        train, val, test = split(cohort, fold=f, folds=a.folds or None)
        tr_dl, va_dl, te_dl = loaders(train, val, test, a.bs)
        model = build(kind, cfg)
        print(f"[{a.role}] fold={f}  params={n_params(model):,}  train windows={len(tr_dl.dataset)}")
        fit(model, tr_dl, va_dl, class_weights(train), a.epochs, teacher=teacher)
        m = evaluate(model, te_dl); results.append(m)
        print(f"  TEST  acc {m['acc']:.3f}  macroF1 {m['macro_f1']:.3f}  kappa {m['kappa']:.3f}  {m['per_class_f1']}")
    if a.folds:
        print("MEAN over folds:", {k: round(float(np.mean([r[k] for r in results])), 3) for k in ("acc", "macro_f1", "kappa")})

    meta = dict(data=a.data, subjects=a.subjects, hours=a.hours, seed=a.seed, test=results[-1], role=a.role)
    save_ckpt(ARTIFACTS / f"{a.role}.pt", model, kind, cfg, meta)
    print(f"saved -> {ARTIFACTS / (a.role + '.pt')}")


if __name__ == "__main__":
    main()
