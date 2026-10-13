"""
Module 5 - Compression:  student.pt  ->  pruned.pt (30% weights zeroed)  ->  int8.pt
   python compress.py
"""
import argparse, copy
import torch, torch.nn as nn
import torch.nn.utils.prune as prune
from config import ARTIFACTS
from registry import load_ckpt, save_ckpt
from train import get_cohort, split, loaders, fit, class_weights, evaluate


def prune_model(model, amount):
    targets = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    for m, name in targets:
        prune.remove(m, name)                       # make the zeros permanent
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amount", type=float, default=0.3)
    ap.add_argument("--finetune", type=int, default=3)
    a = ap.parse_args()

    model, ck = load_ckpt(ARTIFACTS / "student.pt")
    meta = ck["meta"]
    class Args: data, subjects, hours, seed = meta["data"], meta["subjects"], meta["hours"], meta["seed"]
    train, val, test = split(get_cohort(Args))
    tr, va, te = loaders(train, val, test, 16)

    print(f"1) pruning {a.amount:.0%} of conv/linear weights, then fine-tuning {a.finetune} epochs")
    pruned = prune_model(copy.deepcopy(model), a.amount)
    pruned = fit(pruned, tr, va, class_weights(train), a.finetune, lr=5e-4)
    # fine-tuning can regrow zeros, so prune once more and keep the result
    pruned = prune_model(pruned, a.amount)
    m = evaluate(pruned, te); print(f"   pruned  acc {m['acc']:.3f} macroF1 {m['macro_f1']:.3f}")
    save_ckpt(ARTIFACTS / "pruned.pt", pruned, ck["kind"], ck["cfg"], dict(meta, test=m))

    print("2) dynamic INT8 quantization of all Linear layers (attention, FFN, projections)")
    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
    m = evaluate(q, te); print(f"   int8    acc {m['acc']:.3f} macroF1 {m['macro_f1']:.3f}")
    save_ckpt(ARTIFACTS / "int8.pt", q, ck["kind"], ck["cfg"], dict(meta, test=m), quantized=True)


if __name__ == "__main__":
    main()
