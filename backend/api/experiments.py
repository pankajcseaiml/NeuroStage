"""Experiments and benchmark comparison API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.auth.router import get_current_user
from backend.db.database import get_db
from backend.db.models import Experiment, User

router = APIRouter(tags=["experiments"])

BASELINE_BENCHMARKS = [
    {
        "model": "DeepSleepNet",
        "accuracy": 0.820,
        "macro_f1": 0.760,
        "kappa": 0.760,
        "param_count": 25800000,
        "latency_ms": None,
        "model_size_mb": 98.5,
        "source": "Supratak et al., 2017",
        "note": "Published result — not measured in this repository",
    },
    {
        "model": "TinySleepNet",
        "accuracy": 0.826,
        "macro_f1": 0.769,
        "kappa": 0.775,
        "param_count": 26450,
        "latency_ms": None,
        "model_size_mb": 0.9,
        "source": "Supratak & Guo, 2020",
        "note": "Published result — not measured in this repository",
    },
    {
        "model": "AttnSleep",
        "accuracy": 0.837,
        "macro_f1": 0.783,
        "kappa": 0.790,
        "param_count": 3070000,
        "latency_ms": None,
        "model_size_mb": 11.7,
        "source": "Eldele et al., 2021",
        "note": "Published result — not measured in this repository",
    },
]


@router.get("/experiments")
def list_experiments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """List training experiments stored in the database."""
    exps = db.query(Experiment).order_by(Experiment.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": db.query(Experiment).count(),
        "experiments": [
            {
                "id": e.id,
                "mlflow_run_id": e.mlflow_run_id,
                "model_variant": e.model_variant,
                "accuracy": e.accuracy,
                "macro_f1": e.macro_f1,
                "kappa": e.kappa,
                "param_count": e.param_count,
                "latency_ms": e.latency_ms,
                "model_size_mb": e.model_size_mb,
                "created_at": e.created_at.isoformat(),
            }
            for e in exps
        ],
    }


@router.get("/experiments/{experiment_id}")
def get_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get full metrics for one experiment including per-class F1 and confusion matrix."""
    exp = db.get(Experiment, experiment_id)
    if not exp:
        from fastapi import HTTPException
        raise HTTPException(404, "Experiment not found.")
    return {
        "id": exp.id,
        "mlflow_run_id": exp.mlflow_run_id,
        "model_variant": exp.model_variant,
        "accuracy": exp.accuracy,
        "macro_f1": exp.macro_f1,
        "kappa": exp.kappa,
        "param_count": exp.param_count,
        "latency_ms": exp.latency_ms,
        "model_size_mb": exp.model_size_mb,
        "metrics_json": exp.metrics_json,
        "created_at": exp.created_at.isoformat(),
    }


@router.get("/benchmark")
def get_benchmark(db: Session = Depends(get_db)):
    """
    Return accuracy-vs-efficiency comparison:
    NeuroStage variants (from DB) + published baselines (hard-coded from papers).

    IMPORTANT: Baseline values are published results from papers.
    NeuroStage values are measured results from this repository's training runs.
    These must not be mixed without clear labeling in the UI.
    """
    neurostage_experiments = db.query(Experiment).all()
    neurostage_entries = [
        {
            "model": e.model_variant,
            "accuracy": e.accuracy,
            "macro_f1": e.macro_f1,
            "kappa": e.kappa,
            "param_count": e.param_count,
            "latency_ms": e.latency_ms,
            "model_size_mb": e.model_size_mb,
            "source": "This repository (measured)",
            "is_target": False,
        }
        for e in neurostage_experiments
        if e.accuracy is not None
    ]
    return {
        "neurostage_results": neurostage_entries,
        "baselines": BASELINE_BENCHMARKS,
        "note": (
            "Baseline accuracy/F1 values are from published papers. "
            "Latency values for baselines are not measured in this repository. "
            "NeuroStage values are experimentally measured results."
        ),
    }
