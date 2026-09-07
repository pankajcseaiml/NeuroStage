# NeuroStage Production Deployment Guide

This guide details best practices for deploying NeuroStage in containerized production and edge environments.

---

## 1. System Requirements

### Recommended Server Specs
- **CPU**: 4+ cores (x86_64 or ARM64)
- **RAM**: 8 GB minimum (16 GB recommended for batch inference & training)
- **Storage**: 50 GB SSD (PostgreSQL database, MLflow artifacts, EDF uploads)
- **OS**: Ubuntu 22.04 LTS, Debian 12, or container host (Docker Engine 24+)

### Edge Hardware Targets
- **NVIDIA Jetson Orin Nano / Nano**: TensorRT / PyTorch GPU runtime.
- **Raspberry Pi 4 / 5 (8 GB)**: PyTorch ARM64 with QNNPACK INT8 quantization.
- **Embedded ARM Cortex-M**: Requires exported ONNX/TFLite micro model with static INT8 weights.

---

## 2. Docker Compose Production Deployment

The project provides a battle-tested `docker-compose.yml` orchestrating:
1. **PostgreSQL 16**: Relational storage for users, recordings, and experiment logs.
2. **MLflow Tracking Server**: Experiment metrics, parameter logging, and model registry.
3. **FastAPI Backend**: Python 3.11 asynchronous API running with Uvicorn.
4. **Next.js 16 Frontend**: Standalone optimized Node.js SSR production server.

### Step 1: Prepare Environment Secrets
Never deploy with default secret keys. Create a production `.env`:

```bash
# Generate a cryptographically strong 64-character secret key
python -c "import secrets; print(secrets.token_hex(32))"
```

Populate `.env`:
```env
SECRET_KEY=your_generated_cryptographic_secret_here
POSTGRES_USER=neurostage_prod
POSTGRES_PASSWORD=your_secure_db_password
DATABASE_URL=postgresql://neurostage_prod:your_secure_db_password@postgres:5432/neurostage
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_CHECKPOINT=runs/checkpoints/neurostage_base_best.pt
MODEL_CONFIG_PATH=configs/base.yaml
CORS_ORIGINS=["https://neurostage.yourdomain.com"]
NEXT_PUBLIC_API_URL=https://api.neurostage.yourdomain.com
```

### Step 2: Build and Launch Containers
```bash
# Build multi-stage images without cache
docker compose build --no-cache

# Launch in detached mode
docker compose up -d
```

### Step 3: Verify Container Health
```bash
docker compose ps
```
Both backend and PostgreSQL include container health checks that report `healthy` once operational.

---

## 3. Reverse Proxy & SSL/TLS Configuration (Nginx)

In production, terminate SSL/TLS using Nginx or Cloudflare in front of the application services:

```nginx
# API Backend
server {
    listen 443 ssl http2;
    server_name api.neurostage.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.neurostage.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.neurostage.yourdomain.com/privkey.pem;

    client_max_body_size 500M;  # Allow large EDF uploads

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Frontend Dashboard
server {
    listen 443 ssl http2;
    server_name neurostage.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/neurostage.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/neurostage.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 4. Security Hardening Checklist

- [ ] **Unprivileged Container Execution**: Both `Dockerfile.backend` (user `appuser`) and `frontend/Dockerfile` (user `nextjs`) run as non-root users.
- [ ] **Database Network Isolation**: Ensure PostgreSQL port `5432` is not exposed publicly on external interfaces; restrict access to the Docker internal bridge network.
- [ ] **JWT Key Rotation**: Store the active `SECRET_KEY` in your production secret manager (AWS Secrets Manager, HashiCorp Vault, or Kubernetes Secrets).
- [ ] **Uploads Directory Scavenging**: Configure a cron job or background worker to delete processed temporary EDF files in `uploads/` after inference completion.

---

## 5. Monitoring & Observability

- **Liveness Probes**: Monitor `GET https://api.yourdomain.com/health` (returns status `ok` and `model_loaded: true`).
- **Structured Logs**: Backend logs emit timestamps, log levels, and module tags:
  ```
  2026-09-07 14:00:00,123 | INFO | neurostage.api | Inference model loaded from runs/checkpoints/...
  ```
- **MLflow Tracking**: Monitor model evaluation metrics and training history via the dedicated MLflow service (`http://localhost:5000`).
