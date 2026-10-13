# NeuroStage Frontend — Interactive Sleep Staging Dashboard

The Next.js 16 web application for **NeuroStage**, providing real-time Polysomnography analysis, full-night hypnogram visualization, Transformer self-attention interpretability heatmaps, and empirical baseline benchmarking.

---

## Key Features & Route Architecture

* **`/` (Overview):** Landing page with the 7-step signal processing pipeline, core metrics, and architecture breakdown.
* **`/predict` (Inference Studio):** Drag-and-drop EDF/BDF polysomnography upload with JWT authentication and real-time inference triggering.
* **`/hypnogram` (Sleep Timeline):** Interactive multi-stage timeline rendered with Recharts displaying standard AASM sleep stages (Wake, N1, N2, N3, REM), per-epoch prediction confidence, and overall sleep architecture percentages.
* **`/attention` (Explainability Heatmap):** Inspects the Transformer's $20 \times 20$ multi-head self-attention matrix to show which preceding and succeeding epochs drove the classification decision.
* **`/compare` (Benchmark Comparison):** Interactive scatter plot comparing NeuroStage against published baselines (DeepSleepNet, TinySleepNet, AttnSleep) across parameter scale, model footprint, and accuracy.
* **`/experiments` (Audit Log):** Connects to the backend database to view historical MLflow training runs, validation metrics, and per-class F1 scores.

---

## Tech Stack

* **Framework:** Next.js 16.3 (App Router) + React 19
* **Language:** TypeScript 5
* **Data Visualization:** Recharts
* **Styling:** Vanilla CSS Variables + Modern Design System Tokens
* **Communication:** REST APIs targeting the FastAPI backend (`http://localhost:8000`)

---

## Local Development

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env.local
```
Ensure `NEXT_PUBLIC_API_URL` points to your running FastAPI backend:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Run Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
