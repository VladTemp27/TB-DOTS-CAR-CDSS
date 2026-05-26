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

This system is the implementation artifact for a thesis on **Clinical Decision Support for TB-DOTS using Explainable AI**. **[Read the full thesis (PDF)](paper/apa/thesis_apa.pdf)**

---

## Live System

**URL:** [https://dots-cdss.bennygil.me](https://dots-cdss.bennygil.me)

### Demo Credentials

| Field | Value |
|---|---|
| **Username** | `admin123` |
| **Password** | `password123` |

> These are demo credentials for evaluation purposes. The system is intended for authorized DOH-CAR personnel only.

---

## Key Features

| Feature | Description |
|---|---|
| **Risk Stratification** | XGBoost/LightGBM ensemble predicts treatment outcome risk at intake (M0) and each monthly follow-up |
| **Temporal SHAP** | Month-over-month SHAP contribution tracking shows which clinical factors drive risk changes |
| **AI Clinical Narratives** | MedGemma 1.5B local LLM generates plain-language clinical interpretations of risk scores |
| **Patient Registry** | Full patient intake, monthly check-in workflow, and longitudinal profile tracking |
| **X-ray Management** | Upload, store, and view chest X-rays per patient |
| **Offline-first LLM** | Model runs fully on-device — no external API calls for clinical inference |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Browser (PWA)                   │
│          React 18 + Vite + Tailwind CSS          │
│     Pages: Login, Dashboard, Patient Intake,     │
│     Monthly Check-in, Profile, Risk Update       │
└───────────────────┬─────────────────────────────┘
                    │ HTTP / SSE
┌───────────────────▼─────────────────────────────┐
│              FastAPI Backend (Python)            │
│  /api/patients   /api/xrays   /api/ai/explain   │
│                                                  │
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │  ML Models   │  │  MedGemma LLM (llama.cpp)│  │
│  │  XGBoost /   │  │  medgemma-1.5-4b-it      │  │
│  │  LightGBM    │  │  (quantized, on-device)  │  │
│  └──────────────┘  └─────────────────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │           SQLite Database                │   │
│  │  Patients · Monthly Records · Predictions│   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## Local Development

### Prerequisites

- Python 3.12
- Node.js 18+

### 1. Clone & Install

```bash
git clone https://github.com/Benny-Gil/TB-DOTS-CAR-CDSS.git
cd TB-DOTS-CAR-CDSS
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
npm install --prefix web-app
```

### 2. Download the LLM Model

> **Note:** MedGemma is for local development only. The live deployed system does **not** include the model — AI narrative generation is disabled in production.

Download the quantized model (~2.4 GB) from Hugging Face:

```bash
mkdir -p models
wget -O models/medgemma-1.5-4b-it-IQ4_XS.gguf \
  "https://huggingface.co/bartowski/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-IQ4_XS.gguf"
```

> Apple Silicon (M-series) is recommended — the backend enables Metal GPU offload automatically.

### 3. Start Everything

```bash
./dev.sh
```

`dev.sh` handles migrations, seeds demo data on first run, and starts both services with a live terminal dashboard.

| Flag | Effect |
|---|---|
| _(none)_ | Binds to `127.0.0.1` — localhost only |
| `--lan` | Binds to `0.0.0.0` — accessible to other devices on the network |
| `--from-source` | Compiles `llama-cpp-python` from source instead of using a prebuilt wheel |

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
└── paper/apa/            # Thesis manuscript and PDF
```

---

## ML Pipeline

The system uses a **6-configuration grid** (XGBoost / LightGBM / Random Forest × None / SMOTE-ENN) trained on a **599-patient** CAR TB dataset with a **14-feature** clinical feature set.

Risk scores are computed at:
- **M0 (Intake)** — baseline prediction from enrollment features
- **M1–M6** — updated monthly using medication adherence and clinical follow-up data

SHAP values are computed at each time point and displayed as feature contribution bars in the patient profile, with deltas shown month-over-month.

---

## API Reference

The backend exposes a REST + SSE API. Full interactive docs available at `/docs` when running locally.

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

---

<div align="center">

Built for the Department of Health — Cordillera Administrative Region

</div>
