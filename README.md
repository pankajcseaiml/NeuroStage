# NeuroStage 🧠

<div align="center">

**A Lightweight CNN-Transformer Framework for Real-Time Single-Channel EEG Sleep Stage Classification**

[![Backend & Frontend CI](https://github.com/pankajcseaiml/NeuroStage/actions/workflows/ci.yml/badge.svg)](https://github.com/pankajcseaiml/NeuroStage/actions/workflows/ci.yml)
[![CodeQL Security Scan](https://github.com/pankajcseaiml/NeuroStage/actions/workflows/codeql.yml/badge.svg)](https://github.com/pankajcseaiml/NeuroStage/actions/workflows/codeql.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Node.js Version](https://img.shields.io/badge/node-20%2B-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16%20App%20Router-black.svg?logo=next.js&logoColor=white)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SemVer](https://img.shields.io/badge/semver-2.0.0-blue)](https://semver.org/)

</div>

---

## ⚠️ Research Prototype Disclaimer

> [!IMPORTANT]
> **NeuroStage is a research and educational engineering prototype.** It is **not** a clinically certified medical diagnostic device under FDA, CE-MDR, or CDSCO standards. All predicted hypnograms, attention weights, and sleep stage classifications are for research, experimentation, and educational purposes only.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model Design](#model-design)
- [Research Rigor](#research-rigor)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
  - [Docker Compose Deployment](#docker-compose-deployment)
- [Data Pipeline & Training](#data-pipeline--training)
- [Model Compression & Benchmarking](#model-compression--benchmarking)
- [REST API Reference](#rest-api-reference)
- [Testing & Quality Controls](#testing--quality-controls)
- [Security & Zero-Trust Model](#security--zero-trust-model)
- [Community & Contributing](#community--contributing)
- [Authors & Acknowledgments](#authors--acknowledgments)
- [Citation & References](#citation--references)

---

## Overview

Polysomnography (PSG) is the clinical gold standard for diagnosing sleep disorders, but manual scoring of multi-channel recordings across a full night (approx. 1,000 thirty-second epochs) is tedious, time-consuming, and subject to inter-rater variability. While existing deep learning architectures achieve high classification accuracy, they typically rely on heavy recurrent neural networks (RNNs) or large Transformers that require significant compute resources, rendering them impractical for edge deployment on wearable sleep monitors.

**NeuroStage** bridges this gap with a compact, hybrid deep learning framework designed specifically for single-channel raw EEG signals:
1. **Intra-Epoch Waveform Extraction**: Multi-scale 1D Convolutional Neural Networks utilizing **depthwise-separable convolutions** to extract multi-band oscillatory features directly from 30-second raw EEG epochs (3,000 samples @ 100 Hz).
2. **Inter-Epoch Macro-Temporal Modeling**: A compact **Multi-Head Self-Attention Transformer** captures macro-sleep architecture transitions across a 20-epoch (10-minute) contextual sequence.
3. **Model Compression Engine**: Unstructured magnitude pruning and Post-Training Quantization (INT8 PTQ) reduce parameter count and inference latency for wearable hardware.
4. **Production Full-Stack Application**: A high-performance FastAPI service with PostgreSQL persistence, MLflow experiment tracking, and an interactive Next.js 16 dashboard for live hypnogram exploration and attention visualization.

---

## Key Features

- **Single-Channel EEG Input**: Classifies standard AASM sleep stages: **Wake (W)**, **N1 (Light Sleep)**, **N2 (Intermediate Sleep)**, **N3 (Slow-Wave / Deep Sleep)**, and **REM**.
- **Depthwise-Separable Convolutions**: Achieves up to 8–9× reduction in Multiply-Accumulate operations (MACs) compared to standard 1D convolutions.
- **Interpretable Attention Weights**: Exposes Transformer attention maps to visualize which contextual epochs influenced each classification decision.
- **Leak-Free Validation**: Strict subject-independent K-fold cross-validation ensures no data leakage between subjects across training and test splits.
- **Automated Preprocessing**: MNE-Python pipeline for FIR bandpass filtering (0.3–35 Hz), channel extraction (Fpz-Cz / Pz-Oz), resampling, and z-score normalization.
- **Interactive Web Dashboard**: Upload EDF recordings, trigger inference, inspect hypnograms, analyze stage distribution metrics, and review MLflow experiments.
- **Comprehensive DevSecOps**: Hardened container configurations, GitHub Actions CI/CD, CodeQL static code analysis, and Dependabot automation.

---

## System Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Raw Polysomnography (EDF)   │
                                  └───────────────┬───────────────┘
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │   Signal Preprocessing (MNE)  │
                                  │  (0.3-35 Hz Bandpass, 100 Hz) │
                                  └───────────────┬───────────────┘
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  NeuroStage Core Architecture                                    │
│                                                                                                  │
│    30s EEG Epoch (1, 3000)                                                                       │
│               │                                                                                  │
│               ▼                                                                                  │
│    ┌──────────────────────┐                                                                      │
│    │ Multi-Scale 1D-CNN   │──► Branch 1: Kernel 50 (500 ms @ 100 Hz) ── Depthwise-Separable      │
│    │ Feature Extractor    │──► Branch 2: Kernel 25 (250 ms @ 100 Hz) ── Depthwise-Separable      │
│    └──────────┬───────────┘──► Branch 3: Kernel 10 (100 ms @ 100 Hz) ── Depthwise-Separable      │
│               │                                                                                  │
│               ▼                                                                                  │
│    128-dim Epoch Embeddings (Sequence Length = 20 epochs / 10 min)                               │
│               │                                                                                  │
│               ▼                                                                                  │
│    ┌──────────────────────┐                                                                      │
│    │ Compact Transformer  │──► 2 Layers, 4 Attention Heads, Pre-LN                               │
│    │ Temporal Encoder     │──► Sinusoidal Positional Encoding & Residuals                        │
│    └──────────┬───────────┘──► Attention Weights Matrix (20 x 20)                                │
│               │                                                                                  │
│               ▼                                                                                  │
│    ┌──────────────────────┐                                                                      │
│    │ Classification Head  │──► Dense(128) ──► GELU ──► Dropout(0.25) ──► Dense(5) ──► Softmax   │
│    └──────────┬───────────┘                                                                      │
│               │                                                                                  │
│               ▼                                                                                  │
│    Predicted Stage (Wake · N1 · N2 · N3 · REM)                                                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
┌─────────────────────────────────┐                             ┌──────────────────────────────────┐
│      FastAPI REST Backend       │                             │      Next.js 16 Web Dashboard    │
│  - JWT Authentication (Bcrypt)  │◄──────── REST APIs ────────►│  - Live EDF Upload & Predict     │
│  - Asynchronous EDF Processing  │                             │  - Dynamic Hypnogram (Recharts)  │
│  - PostgreSQL + SQLAlchemy ORM  │                             │  - Attention Heatmap Inspector   │
│  - MLflow Experiment Tracking   │                             │  - Model Comparison & Ablation   │
└─────────────────────────────────┘                             └──────────────────────────────────┘
```

---

## Model Design

### 1. Multi-Scale 1D CNN Extractor
The feature extractor operates on single raw 30-second EEG epochs (3,000 samples @ 100 Hz). To capture both slow rhythms (delta 0.5–4 Hz, theta 4–8 Hz) and transient fast events (sleep spindles 11–16 Hz, vertex sharp waves):
- **Branch 1 (Kernel 50 = 500 ms)**: Captures slow delta waveforms and low-frequency spectral envelopes.
- **Branch 2 (Kernel 25 = 250 ms)**: Tuned to intermediate theta and alpha band rhythms.
- **Branch 3 (Kernel 10 = 100 ms)**: Specialized for high-frequency sleep spindles and steep K-complexes.
- **Depthwise-Separable Convolutions**: Replaces standard Conv1d with a spatial depthwise convolution followed by a pointwise $1\times 1$ convolution, drastically lowering FLOPs and parameter count.

### 2. Transformer Temporal Sequence Encoder
Sleep stages do not occur in isolation; physiological transitions follow well-defined macro-temporal architecture. The temporal encoder models context sequences:
- Context Window: 20 consecutive epochs (10 minutes of continuous sleep).
- Architecture: 2 Transformer encoder layers with 4 attention heads each ($d_{\text{model}} = 64$ or $128$).
- Positional Encoding: Sinusoidal positional encodings maintain relative epoch timing within the window.
- Interpretability: Attention maps from the final layer are extracted and exposed through the API for clinical interpretability.

---

## Research Rigor

| Research Standard | Implementation in NeuroStage |
|---|---|
| **Subject-Independent Splits** | K-fold cross-validation strictly partitions data by subject IDs (`neurostage/data/splits.py`). No epochs from the same subject appear in both training and test sets. |
| **Leak-Free Preprocessing** | Normalization and filtering are applied per subject/recording; statistics are never shared across splits. |
| **Class Imbalance Mitigation** | Supports class-weighted cross-entropy loss based on training fold inverse stage frequency (crucial for rare N1 stages). |
| **Comprehensive Metrics** | Evaluates overall Accuracy, Macro-F1, Cohen's Kappa ($\kappa$), and per-class F1 for Wake, N1, N2, N3, and REM. |
| **Experiment Reproducibility** | Deterministic seeding (`set_seed`), YAML configuration tracking, and full MLflow run artifact logging. |

---

## Repository Structure

```
NeuroStage/
├── .github/
│   ├── ISSUE_TEMPLATE/             # Structured GitHub issue forms
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   ├── documentation.yml
│   │   └── config.yml              # Directs security issues to private advisories
│   ├── workflows/                  # Production CI/CD pipelines
│   │   ├── ci.yml                  # Backend CI, Frontend CI, Secret Scanning
│   │   ├── codeql.yml              # Static application security testing (SAST)
│   │   └── dependency-review.yml   # PR dependency vulnerability review
│   ├── CODEOWNERS                  # Repository code ownership configuration
│   ├── dependabot.yml              # Automated dependency update configuration
│   └── pull_request_template.md    # Pull request submission checklist
├── backend/                        # FastAPI application service
│   ├── api/                        # REST endpoint routers (eeg, experiments)
│   ├── auth/                       # JWT authentication and security utilities
│   ├── core/                       # App settings and environment config
│   ├── db/                         # SQLAlchemy database models and sessions
│   ├── services/                   # ML inference orchestration service
│   └── main.py                     # FastAPI application entry point
├── configs/                        # YAML configuration files
│   ├── base.yaml                   # Baseline full CNN-Transformer configuration
│   ├── ablation_cnn_only.yaml      # Ablation: CNN feature extractor only
│   ├── ablation_transformer_1l.yaml# Ablation: 1-layer Transformer
│   ├── ablation_transformer_4l.yaml# Ablation: 4-layer Transformer
│   └── teacher.yaml                # Teacher model config for Knowledge Distillation
├── docs/                           # Technical documentation suite
│   ├── API.md                      # REST API endpoints & schema documentation
│   ├── ARCHITECTURE.md             # Complete UML diagrams & system architecture
│   ├── DEPLOYMENT.md               # Production deployment and container guidance
│   ├── PANEL_PRESENTATION_GUIDE.md # Technical defense guide & panel Q&A playbook
│   ├── PROJECT_TECHNICAL_ARCHITECTURE.md # Stack audit and system specification
│   └── SYNOPSIS.md                 # Department Capstone Phase 1 Synopsis
├── frontend/                       # Next.js 16 + React 19 web dashboard
│   ├── app/                        # Next.js App Router pages
│   │   ├── attention/              # Transformer attention weights explorer
│   │   ├── compare/                # Accuracy-vs-efficiency scatter comparisons
│   │   ├── experiments/            # Sortable MLflow experiment table
│   │   ├── hypnogram/              # Interactive hypnogram visualization
│   │   ├── predict/                # EDF file upload & prediction interface
│   │   ├── layout.tsx              # Root layout & navigation
│   │   └── page.tsx                # Landing & overview page
│   ├── Dockerfile                  # Multi-stage container build for frontend
│   └── package.json                # Frontend package dependencies
├── neurostage/                     # Core PyTorch Machine Learning package
│   ├── compression/                # Pruning, INT8 quantization & distillation
│   ├── data/                       # MNE EDF processing, datasets & K-fold splits
│   ├── evaluation/                 # Metrics, benchmarks & efficiency profiling
│   ├── models/                     # CNN, Transformer & unified NeuroStage model
│   ├── training/                   # Training loop, loss functions & MLflow hooks
│   └── config.py                   # YAML & CLI configuration loader
├── scripts/                        # Automation and utility scripts
│   ├── download_data.py            # PhysioNet Sleep-EDF dataset downloader
│   └── prepare_data.py             # Preprocessing to HDF5 dataset pipeline
├── tests/                          # Automated test suite
│   ├── test_api.py                 # FastAPI integration tests
│   ├── test_compression.py         # Pruning and quantization unit tests
│   ├── test_config.py              # Configuration parsing unit tests
│   └── test_model.py               # Model forward-pass & sanity tests
├── .env.example                    # Backend environment variables template
├── .gitignore                      # Git exclusion rules
├── CHANGELOG.md                    # Release history and version notes
├── CODE_OF_CONDUCT.md              # Contributor Covenant v2.1
├── CONTRIBUTING.md                 # Contributor guidelines
├── Dockerfile.backend              # Hardened non-root Dockerfile for backend
├── docker-compose.yml              # Full-stack Docker Compose service definition
├── LICENSE                         # MIT License
├── pyproject.toml                  # Python package specifications & build tool config
├── README.md                       # Repository landing page
├── compress.py                     # Compression CLI (pruning, INT8)
├── evaluate.py                     # Checkpoint evaluation CLI
└── train.py                        # Model training CLI
```

---

## Quick Start

### Prerequisites
- **Python**: 3.10, 3.11, or 3.13
- **Node.js**: 20+ and npm 10+
- **Docker & Docker Compose**: (Optional, for containerized execution)

---

### Local Development Setup

#### 1. Clone the repository
```bash
git clone https://github.com/pankajcseaiml/NeuroStage.git
cd NeuroStage
```

#### 2. Python Virtual Environment & Backend Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install package in editable mode with development & backend extras
pip install -e ".[backend,dev]"

# Configure environment variables
cp .env.example .env
```

#### 3. Frontend Setup
```bash
cd frontend
npm ci
cp .env.example .env.local
cd ..
```

#### 4. Run the Automated Tests
```bash
# Run complete Python test suite
pytest tests/ -v --tb=short

# Run Python linter
ruff check neurostage/ backend/ tests/ train.py evaluate.py compress.py

# Run Frontend linter
cd frontend && npm run lint && cd ..
```

---

### Docker Compose Deployment

To orchestrate the entire full-stack platform (PostgreSQL, MLflow Tracking Server, FastAPI Backend, and Next.js Frontend) in a single command:

```bash
docker compose up -d
```

| Service | Address | Description |
|---|---|---|
| **Frontend Web App** | [http://localhost:3000](http://localhost:3000) | Interactive dashboard, hypnogram viewer, and prediction UI |
| **FastAPI REST API** | [http://localhost:8000](http://localhost:8000) | API service and backend endpoints |
| **Interactive OpenAPI Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI for exploring and testing API endpoints |
| **MLflow Tracking UI** | [http://localhost:5000](http://localhost:5000) | Training metrics, loss curves, and model artifacts |

To shut down the cluster:
```bash
docker compose down
```

---

## Data Pipeline & Training

### 1. Download Dataset
NeuroStage supports standard open polysomnography datasets including **PhysioNet Sleep-EDF Expanded** (Cassette subset).

```bash
# Download 10 subjects from PhysioNet (~1.5 GB)
python scripts/download_data.py --n_subjects 10
```

### 2. Preprocess to HDF5
Processes raw EDF signals through 0.3–35 Hz bandpass FIR filtering, 100 Hz resampling, and 30-second epoch segmentation:
```bash
python scripts/prepare_data.py
```

### 3. Model Training
```bash
# Train baseline model with 5-fold cross-validation
python train.py

# Run with custom hyperparameters
python train.py training.lr=5e-4 model.transformer.n_layers=4

# Run full ablation study matrix
python train.py --ablation
```

---

## Model Compression & Benchmarking

NeuroStage includes built-in tools to measure the trade-off between model size, inference latency, and classification accuracy:

```bash
# Evaluate a trained checkpoint
python evaluate.py --checkpoint runs/checkpoints/neurostage_base_best.pt --benchmark

# Run magnitude pruning (30% sparsity)
python compress.py --checkpoint runs/checkpoints/neurostage_base_best.pt --method pruning --prune_amount 0.3

# Run INT8 dynamic quantization
python compress.py --checkpoint runs/checkpoints/neurostage_base_best.pt --method int8

# Run full compression suite
python compress.py --checkpoint runs/checkpoints/neurostage_base_best.pt --method all
```

---

## REST API Reference

For full schemas, request payloads, and response structures, see [`docs/API.md`](docs/API.md).

| Method | Endpoint | Description | Authentication |
|---|---|---|---|
| `POST` | `/auth/register` | Register a new user account | Public |
| `POST` | `/auth/login` | Authenticate and obtain JWT access token | Public |
| `GET` | `/health` | Liveness probe and model load status | Public |
| `GET` | `/benchmark` | Accuracy vs. efficiency benchmark data | Public |
| `POST` | `/eeg/upload` | Upload EDF recording for processing | **Bearer JWT** |
| `POST` | `/predict/{recording_id}` | Trigger sleep staging inference | **Bearer JWT** |
| `GET` | `/hypnogram/{recording_id}` | Retrieve classified hypnogram and confidence | **Bearer JWT** |
| `GET` | `/attention/{recording_id}/{epoch}` | Retrieve Transformer self-attention weights | **Bearer JWT** |
| `GET` | `/experiments` | List training runs, hyperparameters, and metrics | **Bearer JWT** |

---

## Testing & Quality Controls

The repository maintains strict quality gates integrated into GitHub Actions CI:

```bash
# Execute Python tests
pytest tests/ -v --tb=short

# Verify Python syntax and code style
ruff check neurostage/ backend/ tests/ train.py evaluate.py compress.py

# Verify Frontend TypeScript & ESLint rules
cd frontend && npm run lint

# Verify Frontend production build
cd frontend && npm run build
```

---

## Security & Zero-Trust Model

NeuroStage adheres to a **Zero-Trust Security Model**:
- **No Committed Secrets**: No passwords, JWT secrets, private keys, or API credentials are ever committed.
- **Environment Isolation**: All configuration is managed via `.env` files derived from `.env.example`.
- **Stateless Authentication**: Protected endpoints require signed JWT bearer tokens with configurable expiration.
- **Unprivileged Containers**: Docker containers run as unprivileged non-root users (`appuser` in backend, `nextjs` in frontend).
- **Vulnerability Reporting**: If you discover a vulnerability, please review our [Security Policy](SECURITY.md) and report it confidentially via GitHub Security Advisories.

---

## Community & Contributing

We welcome contributions! Please refer to:
- [Contributing Guidelines](CONTRIBUTING.md) for branch naming, commit standards, and PR workflows.
- [Code of Conduct](CODE_OF_CONDUCT.md) for community participation standards.
- [Changelog](CHANGELOG.md) for version history and release notes.

---

## Authors & Acknowledgments

**Academic Project (B.Tech Final Year, 2026–2027)**  
*Department of Computer Science and Engineering (Artificial Intelligence & Machine Learning)*  
**Kolhapur Institute of Technology's College of Engineering (Autonomous), Kolhapur**

### Project Team
- **Pankaj Govekar** (PRN: 2324000305 · Lead Maintainer)
- **Anurag Ilake** (PRN: 2324000340)
- **Sandesh Salavi** (PRN: 2324000347)

**Project Guide:** Mr. Rajesh Kumar Singh

---

## Citation & References

1. Supratak, A., et al. "DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG and Deep Transfer Learning." *IEEE TNSRE*, 2017.
2. Supratak, A. and Guo, Y. "TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG." *IEEE EMBC*, 2020.
3. Eldele, E., et al. "An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG." *IEEE TNSRE*, 2021.
4. Vaswani, A., et al. "Attention Is All You Need." *NeurIPS*, 2017.
5. American Academy of Sleep Medicine (AASM). *The AASM Manual for the Scoring of Sleep and Associated Events*, Version 2.6.
6. Kemp, B., et al. "Analysis of a Sleep-wake, Respiration and EEG Data Repository." *PhysioNet*, 2000.

---

<div align="center">
  <sub>Built with ❤️ for reproducible, accessible sleep science research.</sub>
</div>
