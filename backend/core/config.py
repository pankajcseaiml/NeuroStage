"""
Backend configuration via environment variables.
All values are read from the environment (or .env file via Pydantic Settings).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # App
    app_name: str = "NeuroStage API"
    debug: bool = False

    # Auth
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_USE_STRONG_RANDOM_KEY"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # Database
    database_url: str = "postgresql://neurostage:neurostage@localhost:5432/neurostage"

    # ML Model
    model_checkpoint: str = "runs/checkpoints/neurostage_base_best.pt"
    model_config_path: str = "configs/base.yaml"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # File uploads
    upload_dir: str = "uploads/"
    max_upload_size_mb: int = 500

    # CORS (for local dev: allow all; restrict in production)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]


settings = Settings()
