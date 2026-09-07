# Changelog

All notable changes to the **NeuroStage** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Edge deployment target benchmarks (ARM Cortex-M, Raspberry Pi, Jetson Orin Nano).
- Multi-modal support (EOG and EMG channels alongside single-channel EEG).
- Quantization-Aware Training (QAT) pipeline.
- Streaming real-time EEG ingestion via WebSocket.

---

## [0.1.0] - 2026-09-07

### Added
- **Core Architecture**:
  - Multi-scale 1D CNN with depthwise-separable convolutions for intra-epoch feature extraction (`neurostage/models/cnn.py`).
  - 2-layer, 4-head Transformer temporal encoder with sinusoidal positional encodings and Pre-LN (`neurostage/models/transformer.py`).
  - Unified `NeuroStage` model with configurable variants (`neurostage`, `cnn_only`, `teacher`).
- **Data Pipeline**:
  - Automated MNE-Python EDF signal preprocessor with filtering (0.3–35 Hz FIR) and z-score normalization (`neurostage/data/preprocess.py`).
  - Memory-mapped HDF5 `SleepDataset` supporting context windows and data augmentation (`neurostage/data/dataset.py`).
  - Leak-free subject-independent K-fold cross-validation splits (`neurostage/data/splits.py`).
- **Training & Evaluation**:
  - PyTorch training loop with class-weighted Cross-Entropy loss and cosine annealing (`neurostage/training/trainer.py`).
  - MLflow experiment logging for all hyperparameters, training loss, and validation metrics.
  - Comprehensive metrics computation: Accuracy, Macro-F1, Cohen's Kappa, and per-class F1 for Wake, N1, N2, N3, REM (`neurostage/evaluation/metrics.py`).
- **Model Compression**:
  - Global unstructured L1 magnitude pruning with sparsity reporting (`neurostage/compression/pruning.py`).
  - Dynamic INT8 Post-Training Quantization with size and latency reduction measurements (`neurostage/compression/quantization.py`).
- **Backend Services**:
  - High-performance FastAPI REST API (`backend/main.py`).
  - JWT authentication with Bcrypt password hashing (`backend/auth/`).
  - EDF upload, background inference orchestration, and hypnogram generation (`backend/api/`).
- **Frontend Dashboard**:
  - Next.js 16 (App Router) + React 19 interactive user interface (`frontend/`).
  - Interactive hypnogram visualization using Recharts AreaCharts.
  - Transformer attention heatmap explorer.
  - Interactive model comparison scatter plots and experiment log table.
- **DevOps & Security**:
  - Multi-stage Docker configurations (`Dockerfile.backend`, `frontend/Dockerfile`) running as unprivileged non-root users.
  - Full-stack `docker-compose.yml` orchestrating PostgreSQL, MLflow, FastAPI backend, and Next.js frontend.
  - GitHub Actions CI/CD workflows: Backend CI, Frontend CI, Secret scanning, CodeQL static analysis, and Dependabot dependency management.
  - Comprehensive unit and integration test suite (`tests/`).
