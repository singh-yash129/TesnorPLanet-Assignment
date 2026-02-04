# DPF Soot Load Prediction System

**Tensor Planet Internship Assignment**

This project implements an End-to-End Predictive Maintenance Pipeline for Diesel Particulate Filters (DPF). It correctly handles the full lifecycle: synthetic data generation, physical modeling, XGBoost training, and a production-grade FastAPI service.

## 📂 Project Structure

| File | Description |
|------|-------------|
| `data_generator.py` | **Physics-Informed Data Generator**: Simulates 100+ vehicles with realistic soot accumulation/regeneration cycles. |
| `train_model.py` | **Training Pipeline**: Feature engineering, XGBoost training, and artifact versioning. |
| `app.py` | **Inference API**: FastAPI service with input validation, logging, and health checks. |
| `test_suite.py` | **Quality Assurance**: Unit and Integration tests for the pipeline. |
| `TechnicalReport.md` | **Documentation**: Detailed write-up on problem framing, tradeoffs, and MLOps strategy. |
| `.github/workflows/ci.yml` | **CI/CD**: Automated testing pipeline. |

## 🚀 Setup & Installation

### Prerequisites
*   Python 3.9+
*   Docker (Optional, for containerization)

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🛠️ Usage Guide

### Step 1: Generate Synthetic Data
Creates realistic telemetry data with hidden physical patterns (e.g., Passive vs Active Regen).
```bash
python data_generator.py
```
*Output: `data/raw/telemetry.csv`, `maintenance.csv`, `trips.csv`*

### Step 2: Train the Model
Trains an XGBoost Regressor to predict soot load (grams).
```bash
python train_model.py
```
*Output: `models/dpf_model_v1.pkl` with validation metrics.*

### Step 3: Run the API Server
Start the production-ready API.
```bash
uvicorn app:app --reload
```
*   **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Health**: [http://localhost:8000/health](http://localhost:8000/health)

### Step 4: Run Tests
Validate the entire system using Pytest.
```bash
pytest test_suite.py
```

## 📡 API Documentation

### `POST /predict/soot-load`
Returns predicted soot load and maintenance recommendation.

**Request:**
```json
{
  "vehicle_id": "V1001",
  "timestamp": "2023-10-27T10:00:00",
  "engine_rpm": 1800,
  "engine_load_pct": 75.5,
  "exhaust_temp_c": 350.2,
  "diff_pressure_kpa": 4.1,
  "exhaust_flow_rate_kg_h": 450.0
}
```

**Response:**
```json
{
  "vehicle_id": "V1001",
  "predicted_soot_load_grams": 42.5,
  "soot_load_pct": 85.0,
  "recommendation": "WARNING: Schedule Active Regeneration within 24h."
}
```

### `GET /model/info`
Returns current model version and trained features.

## 🐳 Docker Deployment

Build and run the containerized application:

```bash
docker build -t dpf-predictor .
docker run -p 8000:8000 dpf-predictor
```
The API is now accessible at `http://localhost:8000`.

## 🔄 CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration.
*   **Trigger**: Push to `main` branch.
*   **Steps**: Statistics Check -> Data Gen -> Train -> Test.
*   **File**: `.github/workflows/ci.yml`
