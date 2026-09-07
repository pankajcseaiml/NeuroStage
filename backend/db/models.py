"""SQLAlchemy ORM models matching the spec database schema."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from backend.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    recordings: Mapped[list["Recording"]] = relationship(back_populates="user")


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    subject_meta_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String, default="uploaded")  # uploaded | processing | done | error
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="recordings")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="recording")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id"), nullable=False)
    epoch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    attention_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    recording: Mapped["Recording"] = relationship(back_populates="predictions")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    mlflow_run_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    model_variant: Mapped[str] = mapped_column(String, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float)
    macro_f1: Mapped[float | None] = mapped_column(Float)
    kappa: Mapped[float | None] = mapped_column(Float)
    param_count: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    model_size_mb: Mapped[float | None] = mapped_column(Float)
    metrics_json: Mapped[dict | None] = mapped_column(JSON)  # Full per-class F1, confusion matrix etc.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
