# NeuroStage REST API Specification

The NeuroStage backend provides a RESTful API powered by FastAPI for user authentication, EEG recording uploads, asynchronous sleep staging inference, hypnogram retrieval, and experiment tracking.

---

## Base URL & Interactive Documentation

- **Base URL**: `http://localhost:8000` (development) or your configured domain in production.
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc UI**: `http://localhost:8000/redoc`
- **OpenAPI Schema JSON**: `http://localhost:8000/openapi.json`

---

## Authentication & Security

NeuroStage uses JSON Web Tokens (JWT) for user authentication. Protected endpoints require the `Authorization` header with a `Bearer` token:

```http
Authorization: Bearer <your_access_token>
```

Tokens are signed using `HS256` and expire after a configurable duration (default: 24 hours).

---

## Endpoints

### 1. System Health & Status

#### `GET /health`
Liveness and readiness probe for health checking and container orchestration.
- **Auth**: Public
- **Response `200 OK`**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "0.1.0"
}
```

#### `GET /`
Root endpoint.
- **Auth**: Public
- **Response `200 OK`**:
```json
{
  "message": "NeuroStage API — see /docs for the OpenAPI interface."
}
```

---

### 2. Authentication

#### `POST /auth/register`
Create a new user account.
- **Auth**: Public
- **Request Body (`application/json`)**:
```json
{
  "email": "user@example.com",
  "password": "secure_password123"
}
```
- **Responses**:
  - `201 Created`: User successfully registered.
  - `400 Bad Request`: Email already registered.

#### `POST /auth/login`
Authenticate with credentials and obtain a JWT access token.
- **Auth**: Public (OAuth2 password form)
- **Request Body (`application/x-www-form-urlencoded`)**:
  - `username`: `user@example.com`
  - `password`: `secure_password123`
- **Response `200 OK`**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```
- **Response `401 Unauthorized`**: Invalid email or password.

---

### 3. EEG Recordings & Predictions

#### `POST /eeg/upload`
Upload an EDF (European Data Format) polysomnography recording for processing.
- **Auth**: Required (`Bearer JWT`)
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: The `.edf` or `.bdf` file (maximum size 500 MB).
- **Response `201 Created`**:
```json
{
  "recording_id": "c3a2f7e8-49d1-4b72-98e6-94a1b02c8df4",
  "filename": "SC4001E0-PSG.edf",
  "status": "uploaded"
}
```

#### `POST /predict/{recording_id}`
Trigger sleep staging inference on an uploaded recording.
- **Auth**: Required (`Bearer JWT`)
- **Path Parameter**: `recording_id` (string UUID)
- **Response `202 Accepted`**:
```json
{
  "recording_id": "c3a2f7e8-49d1-4b72-98e6-94a1b02c8df4",
  "status": "processing",
  "message": "Inference initiated."
}
```

#### `GET /hypnogram/{recording_id}`
Retrieve the classified hypnogram and confidence scores for each 30-second epoch.
- **Auth**: Required (`Bearer JWT`)
- **Path Parameter**: `recording_id` (string UUID)
- **Response `200 OK`**:
```json
{
  "recording_id": "c3a2f7e8-49d1-4b72-98e6-94a1b02c8df4",
  "n_epochs": 960,
  "status": "done",
  "hypnogram": [
    {
      "epoch_index": 0,
      "time_start_sec": 0,
      "predicted_class": 0,
      "stage_name": "Wake",
      "confidence": 0.9842
    },
    {
      "epoch_index": 1,
      "time_start_sec": 30,
      "predicted_class": 1,
      "stage_name": "N1",
      "confidence": 0.8124
    }
  ]
}
```

#### `GET /attention/{recording_id}/{epoch}`
Retrieve the Multi-Head Self-Attention weights for a specific epoch within its contextual window.
- **Auth**: Required (`Bearer JWT`)
- **Path Parameters**:
  - `recording_id`: UUID
  - `epoch`: Integer epoch index
- **Response `200 OK`**:
```json
{
  "recording_id": "c3a2f7e8-49d1-4b72-98e6-94a1b02c8df4",
  "epoch_index": 10,
  "predicted_class": 2,
  "stage_name": "N2",
  "confidence": 0.9234,
  "attention_weights": {
    "context_epochs": 20,
    "weights": [0.02, 0.04, 0.15, 0.35, 0.20, 0.08, 0.04, 0.02, ...]
  }
}
```

---

### 4. Experiments & Benchmarks

#### `GET /benchmark`
Returns baseline model comparisons vs. NeuroStage results (accuracy, parameters, model size, and inference latency).
- **Auth**: Public
- **Response `200 OK`**:
```json
{
  "neurostage_results": [
    {
      "model": "NeuroStage (Base)",
      "accuracy": 0.842,
      "macro_f1": 0.791,
      "kappa": 0.778,
      "param_count": 142000,
      "latency_ms": 12.4,
      "model_size_mb": 0.57,
      "source": "5-fold CV (Sleep-EDF)"
    }
  ],
  "baselines": [
    {
      "model": "DeepSleepNet",
      "accuracy": 0.820,
      "macro_f1": 0.769,
      "param_count": 21500000,
      "source": "Supratak et al., 2017"
    },
    {
      "model": "TinySleepNet",
      "accuracy": 0.854,
      "macro_f1": 0.805,
      "param_count": 1300000,
      "source": "Supratak & Guo, 2020"
    }
  ]
}
```

#### `GET /experiments`
Lists logged training experiments from MLflow and PostgreSQL.
- **Auth**: Required (`Bearer JWT`)
- **Query Parameters**:
  - `limit`: Integer (default 50, max 100)
  - `offset`: Integer (default 0)
- **Response `200 OK`**:
```json
{
  "total": 5,
  "experiments": [
    {
      "id": "e817a02c-47bb-41a9-8392-5645391f5432",
      "mlflow_run_id": "4b68434a974b497693d25ef90f9eef11",
      "model_variant": "neurostage",
      "accuracy": 0.842,
      "macro_f1": 0.791,
      "kappa": 0.778,
      "param_count": 142000,
      "latency_ms": 12.4,
      "model_size_mb": 0.57,
      "created_at": "2026-09-07T14:00:00Z"
    }
  ]
}
```
