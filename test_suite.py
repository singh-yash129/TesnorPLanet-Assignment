import pytest
from fastapi.testclient import TestClient
from app import app, feature_extraction, TelemetryInput
from datetime import datetime
import pandas as pd
import numpy as np

# Initialize Test Client
client = TestClient(app)

# --- Unit Tests: Feature Engineering ---
def test_feature_extraction_logic():
    # Mock Input
    mock_input = TelemetryInput(
        vehicle_id="V001",
        timestamp=datetime.now(),
        engine_rpm=1500,
        engine_load_pct=50,
        exhaust_temp_c=300,
        diff_pressure_kpa=2.5,
        exhaust_flow_rate_kg_h=100,
        prev_pressure_vals=[2.0, 2.2, 2.4], # Trending up
        prev_temp_vals=[300, 300, 300]
    )
    
    # Mock global model features list (usually loaded from file)
    from app import model_artifacts
    model_artifacts['features'] = [
        'engine_rpm', 'engine_load_pct', 'exhaust_temp_c', 
        'diff_pressure_kpa', 'exhaust_flow_rate_kg_h',
        'exhaust_temp_rolling_mean_12', 'diff_pressure_rolling_mean_12',
        'pressure_flow_ratio', 'diff_pressure_delta'
    ]
    
    df = feature_extraction(mock_input)
    
    # Test 1: Shape
    assert df.shape == (1, 9)
    
    # Test 2: Delta Calculation
    # current (2.5) - last_prev (2.4) should be 0.1
    assert np.isclose(df['diff_pressure_delta'].iloc[0], 0.1)
    
    # Test 3: Rolling Mean
    # Vals: 2.0, 2.2, 2.4, 2.5 -> Mean = 2.275
    assert np.isclose(df['diff_pressure_rolling_mean_12'].iloc[0], 2.275)

# --- Integration Tests: API ---
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_model_info():
    response = client.get("/model/info")
    # Depends if model is loaded globally during test, usually yes over TestClient
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "version" in data
        assert "features" in data

def test_prediction_edge_case_zero_flow():
    # Edge Case: Engine Off / Zero Flow (division by zero protection)
    payload = {
        "vehicle_id": "V_ERR",
        "timestamp": str(datetime.now()),
        "engine_rpm": 0,
        "engine_load_pct": 0,
        "exhaust_temp_c": 25,
        "diff_pressure_kpa": 0,
        "exhaust_flow_rate_kg_h": 0,
        "prev_pressure_vals": [],
        "prev_temp_vals": []
    }
    
    # We expect the API to handle the divide by zero inside feature_extraction
    # Note: This test might fail 503 if model isn't trained/loaded, 
    # but checks input validation at least.
    try:
        response = client.post("/predict/soot-load", json=payload)
        # If model loaded, 200. If not, 503. Either way, shouldn't be 500 crash.
        assert response.status_code in [200, 503]
    except Exception:
        pytest.fail("API crashed on zero flow input")

# To run: pytest test_suite.py