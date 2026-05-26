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

<br/>

<img src="web-app/public/amalgam-logo.png" alt="AMALGAM Team Logo" width="200" />

*Developed by Team AMALGAM*

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

## ML Pipeline

The system uses two complementary ML components, each trained on a separate dataset sourced from DOH-CAR TB treatment records.

### Static Model (Non-Temporal)

| | |
|---|---|
| **Dataset** | 8,876 patient records · 2015–2025 · 23 features |
| **Source** | `dataset/non-temporal/2015-2025-ml-ready.csv` |
| **Configurations** | XGBoost / LightGBM / Random Forest × None / SMOTE-ENN (6 configs) |

The static model predicts treatment outcomes from patient data captured **at the time of enrollment** — before any follow-up information is available. Its purpose is to provide an **immediate intake-level risk signal** so clinicians can flag high-risk patients from day one and prioritize early intervention.

### Temporal Model

| | |
|---|---|
| **Dataset** | 599 patient records · 153 columns · monthly tracking M0–M12 |
| **Source** | `dataset/temporal/combined_complete_dataset.csv` |
| **Features** | 14 engineered features per time step (adherence, doses, clinical vitals) |

The temporal model continuously **refines risk scores as treatment progresses** (M0–M6), incorporating monthly medication adherence data and clinical follow-up records. SHAP values are computed at each time point and displayed as feature contribution bars in the patient profile, with month-over-month deltas to show which factors are driving risk changes over time.

| Time Point | Input | Purpose |
|---|---|---|
| **M0 (Intake)** | Enrollment features | Baseline risk score at registration |
| **M1–M6** | Doses taken, missed doses, updated clinical data | Updated risk score per monthly check-in |

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
├── static-model/         # Non-temporal outcome classifier (research)
├── models/               # Trained model artifacts (ONNX / pickle)
├── slm_shap_pipeline/    # Offline SHAP faithfulness benchmark
├── evaluation/           # SLM explanation evaluation tools
├── data/                 # SQLite DB + X-ray storage (gitignored)
└── paper/apa/            # Thesis manuscript and PDF
```

---

## Local Development

### Prerequisites

- Python 3.12
- Node.js 18+

### 1. Clone the Repository

```bash
git clone https://github.com/Benny-Gil/TB-DOTS-CAR-CDSS.git
cd TB-DOTS-CAR-CDSS
```

### 2. (Optional) Download the LLM Model

> MedGemma is used for AI-generated clinical narratives. It is **optional** — the system runs fully without it; only the AI explanation feature will be unavailable. The live deployed system does not include the model.

```bash
mkdir -p models
wget -O models/medgemma-1.5-4b-it-IQ4_XS.gguf \
  "https://huggingface.co/bartowski/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-IQ4_XS.gguf"
```

> Apple Silicon (M-series) is recommended — the backend enables Metal GPU offload automatically.

### 3. Start Everything

Two equivalent entrypoints are available — both handle virtualenv creation, dependency installation, database migrations, demo data seeding, and launching both services:

```bash
# Python TUI dashboard (recommended)
python dev.py

# Bash fallback
./dev.sh
```

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
