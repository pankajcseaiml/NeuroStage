"""Build / save / load any NeuroStage-family model from one checkpoint format."""
import torch
from model import NeuroStage, DEFAULT_CFG
from baselines import DeepSleepNetLite


def build(kind, cfg=None):
    if kind == "baseline":
        return DeepSleepNetLite()
    return NeuroStage(**{**DEFAULT_CFG, **(cfg or {})})


def n_params(model, nonzero=False):
    return sum((p != 0).sum().item() if nonzero else p.numel() for p in model.parameters())


def save_ckpt(path, model, kind, cfg=None, meta=None, quantized=False):
    d = {"kind": kind, "cfg": cfg, "meta": meta or {}, "quantized": quantized}
    d["model"] = model if quantized else None          # INT8 modules are pickled whole
    d["state"] = None if quantized else model.state_dict()
    torch.save(d, path)


def load_ckpt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d["quantized"]:
        model = d["model"]
    else:
        model = build(d["kind"], d["cfg"])
        model.load_state_dict(d["state"])
    return model.eval(), d
