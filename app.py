from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dpf_api")

# --- Pydantic Schemas for Input Validation ---
class TelemetryInput(BaseModel):
    vehicle_id: str
    timestamp: datetime
    engine_rpm: float
    engine_load_pct: float
    exhaust_temp_c: float
    diff_pressure_kpa: float
    exhaust_flow_rate_kg_h: float
    
    # Optional history for rolling windows (simplified for API demo)
    # In real prod, we might fetch history from a Feature Store (e.g. Redis)
    prev_pressure_vals: Optional[List[float]] = [] 
    prev_temp_vals: Optional[List[float]] = []

class PredictionOutput(BaseModel):
    vehicle_id: str
    predicted_soot_load_grams: float
    soot_load_pct: float
    recommendation: str
    confidence_interval: Optional[List[float]] = None

# Global model store
model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup and clean up on shutdown"""
    try:
        # In prod, this path would be an S3 URL or mounted volume
        path = 'models/dpf_model_v1.pkl' 
        if os.path.exists(path):
            artifacts = joblib.load(path)
            model_artifacts['model'] = artifacts['model']
            model_artifacts['features'] = artifacts['features']
            model_artifacts['meta'] = artifacts.get('version', 'unknown')
            print("Model loaded successfully.")
        else:
            print("WARNING: Model file not found. API will return errors for predictions.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield
    
    # Clean up resources
    model_artifacts.clear()

# --- Application Setup ---
app = FastAPI(
    title="DPF Soot Load Predictor",
    description="Predictive Maintenance API for Tensor Planet Internship",
    version="1.0.0",
    lifespan=lifespan
)

# --- Helper Logic ---
def feature_extraction(input_data: TelemetryInput):
    """
    Replicates the logic in train_model.py for a single instance.
    Real-world note: This code duplication is bad. Usually, we use a shared library.
    """
    # Calculate simple rolling means based on provided history
    # If no history provided, fallback to current value (Cold start problem handled)
    p_hist = input_data.prev_pressure_vals + [input_data.diff_pressure_kpa]
    t_hist = input_data.prev_temp_vals + [input_data.exhaust_temp_c]
    
    # Calculate features
    feats = {
        'engine_rpm': input_data.engine_rpm,
        'engine_load_pct': input_data.engine_load_pct,
        'exhaust_temp_c': input_data.exhaust_temp_c,
        'diff_pressure_kpa': input_data.diff_pressure_kpa,
        'exhaust_flow_rate_kg_h': input_data.exhaust_flow_rate_kg_h,
        'exhaust_temp_rolling_mean_12': np.mean(t_hist[-12:]), # Mean of last 12 or fewer
        'diff_pressure_rolling_mean_12': np.mean(p_hist[-12:]),
        'pressure_flow_ratio': input_data.diff_pressure_kpa / (input_data.exhaust_flow_rate_kg_h + 1e-5),
        'diff_pressure_delta': p_hist[-1] - p_hist[-2] if len(p_hist) > 1 else 0.0
    }
    
    # Convert to DataFrame ensuring column order matches training
    df = pd.DataFrame([feats])
    # Ensure all columns exist (in case of schema mismatch)
    for col in model_artifacts['features']:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[model_artifacts['features']]

def generate_recommendation(soot_grams):
    # Thresholds (Assumptions based on generated data range 0-50g)
    MAX_CAPACITY = 50.0
    load_pct = (soot_grams / MAX_CAPACITY) * 100
    
    if load_pct > 90:
        return "CRITICAL: Stop Vehicle immediately. Forced Regen required.", load_pct
    elif load_pct > 75:
        return "WARNING: Schedule Active Regeneration within 24h.", load_pct
    elif load_pct > 60:
        return "NOTICE: Monitor closely. Ensure highway driving if possible.", load_pct
    else:
        return "HEALTHY: Normal operation.", load_pct

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "model_loaded": 'model' in model_artifacts,
        "model_version": model_artifacts.get('meta', 'N/A')
    }

@app.get("/model/info")
def model_info():
    if 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "version": model_artifacts.get('meta', 'unknown'),
        "features": model_artifacts.get('features', []),
        "description": "XGBoost Regressor for DPF Soot Load Prediction"
    }

@app.post("/predict/soot-load", response_model=PredictionOutput)
def predict_single(data: TelemetryInput):
    if 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Feature Engineering
        features_df = feature_extraction(data)
        
        # Inference
        prediction = model_artifacts['model'].predict(features_df)[0]
        prediction = max(0.0, float(prediction)) # Clamp negative predictions
        
        rec_text, load_pct = generate_recommendation(prediction)
        
        return {
            "vehicle_id": data.vehicle_id,
            "predicted_soot_load_grams": round(prediction, 2),
            "soot_load_pct": round(load_pct, 1),
            "recommendation": rec_text,
            "confidence_interval": [round(prediction * 0.9, 2), round(prediction * 1.1, 2)] # Mock CI
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(data_list: List[TelemetryInput]):
    results = []
    for item in data_list:
        try:
            res = predict_single(item)
            results.append(res)
        except Exception:
            results.append({"vehicle_id": item.vehicle_id, "error": "Prediction failed"})
    return {"batch_results": results}