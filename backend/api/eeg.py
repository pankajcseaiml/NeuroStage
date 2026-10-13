"""EEG upload + predict API routers."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.auth.router import get_current_user
from backend.core.config import settings
from backend.db.database import get_db
from backend.db.models import Prediction, Recording, User
from backend.services import inference as inf_svc

router = APIRouter(tags=["eeg"])


@router.post("/eeg/upload", status_code=status.HTTP_201_CREATED)
async def upload_eeg(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload an EEG EDF file. Returns a recording_id."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_size_mb}MB.")

    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".edf", ".bdf"):
        raise HTTPException(400, "Only EDF/BDF files are accepted.")

    # Save to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    recording_id = str(uuid.uuid4())
    saved_path = upload_dir / f"{recording_id}{suffix}"
    with open(saved_path, "wb") as f:
        f.write(content)

    # Create DB record
    recording = Recording(
        id=recording_id,
        user_id=current_user.id,
        filename=file.filename,
        subject_meta_json={"file_size_bytes": len(content)},
        status="uploaded",
    )
    db.add(recording)
    db.commit()
    return {"recording_id": recording_id, "filename": file.filename, "status": "uploaded"}


def _run_inference_job(recording_id: str, edf_path: str, db_url: str) -> None:
    """Background task: preprocess EDF and save predictions to DB."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        recording = db.get(Recording, recording_id)
        if not recording:
            return
        recording.status = "processing"
        db.commit()

        epochs = inf_svc.preprocess_edf_for_inference(edf_path)
        if epochs is None:
            recording.status = "error"
            db.commit()
            return

        result = inf_svc.predict_epochs(epochs)

        # Persist predictions with proper context-window attention extraction
        from backend.services.inference import _cfg
        context_len = _cfg.data.context_epochs if _cfg else 20
        raw_attns = result.get("attention_weights") or []
        for pred in result["predictions"]:
            idx = pred["epoch_index"]
            attn = None
            if raw_attns:
                win_idx = idx // context_len
                pos_in_win = idx % context_len
                if win_idx < len(raw_attns):
                    win_matrix = raw_attns[win_idx]
                    if isinstance(win_matrix, list) and pos_in_win < len(win_matrix):
                        epoch_vec = win_matrix[pos_in_win]
                        attn = {
                            "weights": [epoch_vec],
                            "epoch_weights": epoch_vec,
                            "context_epochs": len(epoch_vec),
                        }
            p = Prediction(
                recording_id=recording_id,
                epoch_index=idx,
                predicted_class=pred["predicted_class"],
                confidence=pred["confidence"],
                attention_json=attn,
            )
            db.add(p)

        recording.status = "done"
        db.commit()


@router.post("/predict/{recording_id}")
async def predict(
    recording_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Trigger inference on an uploaded recording. Returns a job status."""
    if not inf_svc.is_model_loaded():
        raise HTTPException(503, "Model not loaded. Train a model first.")

    recording = db.get(Recording, recording_id)
    if not recording or recording.user_id != current_user.id:
        raise HTTPException(404, "Recording not found.")
    if recording.status == "processing":
        return {"status": "already_processing", "recording_id": recording_id}

    # Find the uploaded file
    upload_dir = Path(settings.upload_dir)
    edf_files = list(upload_dir.glob(f"{recording_id}.*"))
    if not edf_files:
        raise HTTPException(404, "EDF file not found on disk.")

    background_tasks.add_task(
        _run_inference_job,
        recording_id,
        str(edf_files[0]),
        settings.database_url,
    )
    return {"status": "processing_started", "recording_id": recording_id}


@router.get("/hypnogram/{recording_id}")
def get_hypnogram(
    recording_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the full night's predicted sleep stages for charting."""
    recording = db.get(Recording, recording_id)
    if not recording:
        raise HTTPException(404, "Recording not found.")
    if recording.status != "done":
        return {"status": recording.status, "hypnogram": None}

    predictions = (
        db.query(Prediction)
        .filter(Prediction.recording_id == recording_id)
        .order_by(Prediction.epoch_index)
        .all()
    )

    stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    hypnogram = [
        {
            "epoch_index": p.epoch_index,
            "time_start_sec": p.epoch_index * 30,
            "predicted_class": p.predicted_class,
            "stage_name": stage_names[p.predicted_class],
            "confidence": p.confidence,
        }
        for p in predictions
    ]
    return {
        "recording_id": recording_id,
        "n_epochs": len(hypnogram),
        "status": "done",
        "hypnogram": hypnogram,
    }


@router.get("/attention/{recording_id}/{epoch_index}")
def get_attention(
    recording_id: str,
    epoch_index: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return attention weights for a specific epoch."""
    recording = db.get(Recording, recording_id)
    if not recording:
        raise HTTPException(404, "Recording not found.")

    pred = (
        db.query(Prediction)
        .filter(
            Prediction.recording_id == recording_id,
            Prediction.epoch_index == epoch_index,
        )
        .first()
    )
    if not pred:
        raise HTTPException(404, "Prediction not found for this epoch.")

    attn_data = pred.attention_json or {}
    weights_row = attn_data.get("epoch_weights")
    if not weights_row and "weights" in attn_data:
        w = attn_data["weights"]
        if isinstance(w, list) and len(w) > 0:
            if isinstance(w[0], list):
                pos = epoch_index % 20
                weights_row = w[pos] if pos < len(w) else w[0]
            elif isinstance(w[0], (int, float)):
                weights_row = w

    return {
        "epoch_index": epoch_index,
        "predicted_class": pred.predicted_class,
        "confidence": pred.confidence,
        "attention_weights": {
            "weights": [weights_row] if weights_row else None,
            "epoch_weights": weights_row,
            "context_epochs": len(weights_row) if weights_row else 20,
        } if weights_row else None,
    }
