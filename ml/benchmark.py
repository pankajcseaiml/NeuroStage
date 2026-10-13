"""
Accuracy-vs-efficiency benchmark -> artifacts/benchmark.json  (shown on the web dashboard).
Every model is measured the same way on the same held-out subjects:
  latency = time to process ONE new 30 s epoch in streaming mode, 1 CPU thread
            (CNN on the new epoch + temporal model over the last 10 embeddings).
"""
import json, time
import torch
from config import ARTIFACTS, SEQ_LEN, EPOCH_LEN
from registry import load_ckpt, n_params, build
from train import get_cohort, split, loaders, evaluate

ENTRIES = [("baseline", "DeepSleepNet-style (CNN+BiLSTM)"), ("teacher", "NeuroStage-Teacher (large)"),
           ("student", "NeuroStage (FP32)"), ("pruned", "NeuroStage + 30% pruning"),
           ("int8", "NeuroStage + INT8")]


@torch.no_grad()
def latency_ms(model, runs=40):
    torch.set_num_threads(1)
    ep, buf = torch.randn(1, EPOCH_LEN), torch.randn(1, SEQ_LEN, model.embed(torch.randn(1, EPOCH_LEN)).shape[-1])
    for _ in range(5):
        model.temporal(buf)
    t = time.perf_counter()
    for _ in range(runs):
        e = model.embed(ep)
        model.temporal(torch.cat([buf[:, 1:], e.unsqueeze(1)], 1))
    return (time.perf_counter() - t) / runs * 1000


def main():
    rows, test_dl = [], None
    for key, label in ENTRIES:
        path = ARTIFACTS / f"{key}.pt"
        if not path.exists():
            print(f"skip {key} (not trained yet)"); continue
        model, ck = load_ckpt(path)
        if test_dl is None:
            meta = ck["meta"]
            class Args: data, subjects, hours, seed = meta["data"], meta["subjects"], meta["hours"], meta["seed"]
            _, _, test = split(get_cohort(Args))
            test_dl = loaders(test, test, test, 16)[2]
        m = evaluate(model, test_dl)
        size = path.stat().st_size / 1e6
        # INT8 modules store packed weights, so count the parameters of the same architecture instead
        ref = build(ck["kind"], ck["cfg"]) if ck["quantized"] else model
        rows.append(dict(key=key, name=label, params=n_params(ref), nonzero_params=n_params(ref, True) if not ck["quantized"] else n_params(ref),
                         size_mb=round(size, 3), weight_bits=8 if ck["quantized"] else 32, latency_ms=round(latency_ms(model), 2),
                         acc=round(m["acc"], 4), macro_f1=round(m["macro_f1"], 4), kappa=round(m["kappa"], 4),
                         per_class_f1=m["per_class_f1"], data=ck["meta"].get("data")))
        print(f"{label:34s} params {rows[-1]['params']:>9,}  {rows[-1]['latency_ms']:6.1f} ms  "
              f"acc {rows[-1]['acc']:.3f}  F1 {rows[-1]['macro_f1']:.3f}  size {size:.2f} MB")
    (ARTIFACTS / "benchmark.json").write_text(json.dumps(dict(rows=rows), indent=2))
    print("wrote artifacts/benchmark.json")


if __name__ == "__main__":
    main()
