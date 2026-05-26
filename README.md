<div align="center">

<img src="web-app/public/logo.png" alt="TB-DOTS CAR CDSS Logo" width="80" />

# TB-DOTS CAR CDSS

**Clinical Decision Support System for Tuberculosis DOTS Program**
Department of Health — Cordillera Administrative Region

[![Live System](https://img.shields.io/badge/Live%20System-dots--cdss.bennygil.me-22c55e?style=flat-square&logo=globe)](https://dots-cdss.bennygil.me)
[![Paper](https://img.shields.io/badge/Paper-Read%20Thesis-orange?style=flat-square&logo=adobeacrobatreader)](paper/apa/thesis_apa.pdf)
[![Backend](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)](https://dots-cdss.bennygil.me)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb?style=flat-square&logo=react)](https://dots-cdss.bennygil.me)
[![AI](https://img.shields.io/badge/LLM-MedGemma%20(local%20only)-4285F4?style=flat-square&logo=google)](https://huggingface.co/bartowski/medgemma-1.5-4b-it-GGUF)

</div>

---

## Overview

TB-DOTS CAR CDSS is an AI-powered clinical decision support system for managing tuberculosis patients under the DOTS (Directly Observed Treatment, Short-course) program in the Cordillera Administrative Region. It combines **machine learning risk prediction**, **SHAP-based explainability**, and **MedGemma LLM-generated clinical narratives** to assist healthcare workers in monitoring and treating TB patients.

### Key Features

| Feature | Description |
|---|---|
| **Risk Stratification** | XGBoost/LightGBM ensemble predicts treatment outcome risk at intake (M0) and each monthly follow-up |
| **Temporal SHAP** | Month-over-month SHAP contribution tracking shows which clinical factors drive risk changes |
| **AI Clinical Narratives** | MedGemma 1.5B local LLM generates plain-language clinical interpretations of risk scores |
| **Patient Registry** | Full patient intake, monthly check-in workflow, and longitudinal profile tracking |
| **X-ray Management** | Upload, store, and view chest X-rays per patient |
| **Offline-first LLM** | Model runs fully on-device — no external API calls for clinical inference |

---

## Live System

> **URL:** [https://dots-cdss.bennygil.me](https://dots-cdss.bennygil.me)

### Demo Credentials

| Field | Value |
|---|---|
| **Username** | `admin123` |
| **Password** | `password123` |

> These are demo credentials for evaluation purposes. The system is intended for authorized DOH-CAR personnel only.

---

## Local Development

### Prerequisites

- Python 3.12
- Node.js 18+

### 1. Clone & Install

```bash
git clone https://github.com/Benny-Gil/TB-DOTS-CAR-CDSS.git
cd TB-DOTS-CAR-CDSS

# Install all dependencies (Python venv + pip + npm)
just install
```

### 2. Download the LLM Model

> **Note:** The MedGemma LLM is used for local development only. The live deployed system at [dots-cdss.bennygil.me](https://dots-cdss.bennygil.me) does **not** include the model — AI narrative generation is disabled in production.

Download the quantized model (~2.4 GB) from Hugging Face:

```
https://huggingface.co/bartowski/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-IQ4_XS.gguf
```

Or via `wget` / `curl`:

```bash
mkdir -p models
wget -O models/medgemma-1.5-4b-it-IQ4_XS.gguf \
  "https://huggingface.co/bartowski/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-IQ4_XS.gguf"
```

Place the file at:

```
models/medgemma-1.5-4b-it-IQ4_XS.gguf
```


### 3. Seed Demo Data

```bash
python -m backend.seed_demo
```

To include demo X-ray files:

```bash
python -m backend.seed_demo --include-xrays
```

### 4. Start Everything

```bash
./dev.py
```

`dev.sh` installs missing dependencies, runs database migrations, seeds demo data if the database is empty, then starts both services with a live dashboard.

| Flag | Effect |
|---|---|
| _(none)_ | Binds to `127.0.0.1` (localhost only) |
| `--lan` | Binds to `0.0.0.0` so other devices on the network can connect |
needed for unsupported Python versions) |

```bash
# LAN access (other devices on same network)
./dev.sh --lan
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## Project Structure

```
TB-DOTS-CAR-CDSS/
├── backend/              # FastAPI backend
│   ├── main.py           # App entry point
│   ├── models.py         # SQLAlchemy ORM models
│   ├── inference.py      # LLM streaming inference
│   ├── llm.py            # MedGemma loader (llama.cpp)
│   ├── routers/          # API route handlers
│   │   ├── patients.py   # Patient & monthly record endpoints
│   │   ├── xrays.py      # X-ray upload/retrieval
│   │   └── ai.py         # SHAP + LLM explanation endpoint
│   └── seed_demo.py      # Deterministic demo dataset seeder
├── web-app/              # React frontend
│   └── src/
│       ├── pages/        # Login, Dashboard, Intake, Profile…
│       ├── components/   # Shared UI components
│       └── lib/auth.ts   # Session-based auth
├── models/               # ML model artifacts (ONNX / pickle)
├── slm_shap_pipeline/    # Offline SHAP faithfulness benchmark
├── evaluation/           # SLM explanation evaluation tools
├── data/                 # SQLite DB + X-ray storage (gitignored)
├── process-compose.yaml  # Multi-process dev orchestration
└── justfile              # Task runner recipes
```

---

## ML Pipeline

The system uses a **6-configuration grid** (XGBoost / LightGBM / Random Forest × None / SMOTE-ENN) trained on a **599-patient** CAR TB dataset with a **14-feature** clinical feature set.

Risk scores are computed at:
- **M0 (Intake)** — baseline prediction from enrollment features
- **M1–M6** — updated monthly using medication adherence and clinical follow-up data

SHAP values are computed at each time point and displayed as feature contribution bars in the patient profile, with deltas shown month-over-month.

---

## Agent / API Reference

The backend exposes a REST + SSE API suitable for agent integration.

### Key Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/patients` | List all patients |
| `POST` | `/api/patients` | Register a new patient |
| `GET` | `/api/patients/{id}` | Get patient profile |
| `POST` | `/api/patients/{id}/temporal-risk-record` | Save monthly check-in |
| `POST` | `/api/xrays/{patient_id}` | Upload X-ray |
| `GET` | `/api/xrays/{patient_id}` | List patient X-rays |
| `POST` | `/api/ai/explain` | Stream LLM clinical narrative (SSE) |

Full interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Research Context

This system is the implementation artifact for a thesis on **Clinical Decision Support for TB-DOTS using Explainable AI** at the Cordillera Administrative Region.

**[Read the full thesis (PDF)](paper/apa/thesis_apa.pdf)**

The repository also contains:

- `slm_shap_pipeline/` — offline SLM faithfulness-to-SHAP benchmark tooling
- `evaluation/slm_shap_faithfulness/` — quantitative evaluation of LLM explanation fidelity
- `paper/` — thesis manuscript drafts

---

<div align="center">

Built for the Department of Health — Cordillera Administrative Region

</div>
