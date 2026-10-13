"""
FastAPI service that the Node/Express server talks to.     uvicorn api:app --port 8000
"""
import asyncio, json, os, uuid
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config import ARTIFACTS, RECORDINGS, EPOCH_LEN
from inference import Stager, StreamingStager, load_recording, pick_model
from synthetic import generate_night
from preprocess import to_epochs

app = FastAPI(title="NeuroStage ML service")
stager = None


@app.on_event("startup")
def _load():
    global stager
    path = Path(os.environ["NEUROSTAGE_MODEL"]) if os.environ.get("NEUROSTAGE_MODEL") else pick_model()
    stager = Stager(path)
    print("loaded model:", stager.info)


def _save(sig, labels):
    rec_id = uuid.uuid4().hex
    np.savez_compressed(RECORDINGS / f"{rec_id}.npz", signal=sig, **({} if labels is None else {"labels": labels}))
    return rec_id


def _load_rec(rec_id):
    p = RECORDINGS / f"{rec_id}.npz"
    if not p.exists() or not rec_id.isalnum():
        raise HTTPException(404, "recording not found")
    z = np.load(p); return z["signal"], (z["labels"] if "labels" in z.files else None)


@app.get("/health")
def health():
    return dict(ok=True, model=stager.info)


@app.post("/predict")
async def predict(recording: UploadFile = File(...), fs: float = Form(100)):
    data = await recording.read()
    if len(data) > 60_000_000:
        raise HTTPException(413, "File too large (60 MB max)")
    try:
        sig, labels = load_recording(data, recording.filename or "x.npy", fs)
        res = stager.predict_night(sig, labels)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return dict(rec_id=_save(sig, labels), **res)


@app.post("/demo")
def demo(hours: float = 2.0, seed: int = 0):
    hours = min(max(hours, 0.25), 10)
    sig, labels = generate_night(hours, seed or int(np.random.randint(1, 10**6)))
    return dict(rec_id=_save(sig, labels), **stager.predict_night(sig, labels))


@app.get("/explain/{rec_id}/{epoch}")
def explain(rec_id: str, epoch: int):
    sig, _ = _load_rec(rec_id)
    try:
        return stager.explain_epoch(sig, epoch)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/stream/{rec_id}")
async def stream(rec_id: str, eps: float = 5.0):
    """Server-Sent Events: replays the recording epoch by epoch through the REAL-TIME predictor."""
    sig, _ = _load_rec(rec_id)
    epochs = to_epochs(sig.astype(np.float32))
    live = StreamingStager(stager)

    async def gen():
        for i, ep in enumerate(epochs):
            yield f"data: {json.dumps(dict(epoch=i, total=len(epochs), **live.push(ep)))}\n\n"
            await asyncio.sleep(1 / max(eps, 0.1))
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/benchmark")
def benchmark():
    p = ARTIFACTS / "benchmark.json"
    if not p.exists():
        raise HTTPException(404, "Run `python benchmark.py` first")
    return json.loads(p.read_text())
