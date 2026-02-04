# DPF Soot Load Prediction: Technical Report



## 1. Data Generation & Engineering Approach

### 1.1 Synthetic Data Strategy
Since real DPF data wasn't provided, I built a physics-informed generator (`data_generator.py`).
*   **The Physics Model**: Soot accumulation is non-linear. It increases with `engine_load` (dirty combustion) and decreases when `exhaust_temp` > 300°C (passive regeneration).
*   **The Sensor Proxy**: The primary indicator, `differential_pressure`, is confounded by `exhaust_flow`. My generator simulates this relationship ($P = R \times Q$), forcing the ML model to learn the underlying resistance rather than memorizing raw pressure values.

### 1.2 Feature Engineering Choices
Raw telemetry is noisy, so I designed features that isolate filter resistance:
*   **Rolling Means (15m, 1h)**: Smooths out sensor spikes caused by bumps or sudden acceleration.
*   **Pressure/Flow Ratio**: Approximates physical filter resistance, a cleaner proxy for soot mass than raw pressure.
*   **Differential Pressure Delta**: Captures the rate of clogging over time.

### 1.3 Data Versioning Strategy
To handle temporal data shifts and ensure reproducibility:
*   **Raw Data**: Stored in S3 with partition keys `year/month/day/vehicle_id`. Immutable.
*   **Feature Store**: "Time Travel" capability (e.g., using Feast) to fetch feature values exactly as they were at inference time for training, preventing data leakage.
*   **Artifacts**: Models tagged with semantic versioning (v1.0.0) and linked to specific training data snapshots (Git commit hash + Data hash).

---

## 2. Modeling Rationale

### 2.1 Problem Framing
I framed this as a **Regression Problem** (Target: Soot Load in grams).
*   *Why not classification?* A binary "Regen Needed" label loses information. Predicting the actual load allows dynamic thresholds (e.g., "Regen at 80% if near a service center, else push to 90%").

### 2.2 Model Selection: XGBoost
I chose XGBoost (Gradient Boosted Trees) over Linear Regression or Deep Learning:
*   **Non-Linearity**: Handles the complex relationship between RPM, Temp, and Pressure.
*   **Robustness**: Tree models handle outliers and unscaled data better than Neural Networks.
*   **Interpretability**: Feature importance scores help diagnose which sensors drive predictions.

### 2.3 Business Tradeoffs (Precision vs. Recall)
In Predictive Maintenance, costs are asymmetric:
*   **False Negative (Missed Breakdown)**: High cost (Towing, engine damage).
*   **False Positive (Early Regen)**: Low cost (Small fuel penalty).
*   **Strategy**: Optimize for **Recall**. I implemented a tiered warning system: "Warning" at 75% load provides a safety buffer before the 90% "Critical" level.

---

## 3. Production & MLOps Architecture

### 3.1 API Design
The solution is served via **FastAPI**:
*   **Single Inference**: Real-time dashboard updates (`POST /predict/soot-load`).
*   **Batch Inference**: End-of-day fleet analysis (`POST /predict/batch`).
*   **Cold Start**: The feature extractor handles missing history by using instantaneous values, preventing crashes on new vehicles.

### 3.2 Robustness & Quality Assurance
*   **Input Validation**: Pydantic schemas enforce strict data typing.
*   **Zero-Division Protection**: Explicit handling for `exhaust_flow = 0` (engine off) scenarios.
*   **Unit Tests**: `test_suite.py` verifies feature logic and API stability.

### 3.3 CI/CD & Automated Validation
*   **Pipeline**: GitHub Actions triggers on every commit.
*   **Automated Checks**:
    1.  **Code Quality**: Linting and Unit Tests.
    2.  **Model Validation**: The pipeline trains a candidate model and asserts `RMSE < Threshold` on a holdout set before allowing a merge.

### 3.4 Handling Edge Cases & Failures (Bonus)
*   **Sensor Failure**: If `diff_pressure` is missing, the API falls back to a "Safe Mode" prediction based on Engine Load/RPM history, logging a `SENSOR_FAULT`.
*   **Out-of-Range**: Inputs are clamped to physical limits; Pydantic rejects impossible values (e.g., negative RPM).
*   **Stale Data**: Telemetry > 5 minutes old is flagged and excluded from real-time alerts.

---

## 4. Recommendations & Impact

### 4.1 Deployment Strategy
Deploy as a Docker container to Kubernetes. Use a **Sidecar Pattern** to sync model artifacts from S3, allowing model updates without redeploying the application code.

### 4.2 Future Improvements
*   **Integrate "Time Since Last Regen"**: Joining Maintenance tables with Telemetry in a production Feature Store would significantly improve accuracy.
*   **Feedback Loop**: Capture actual regeneration events to retrain the model. If the model predicts 90% soot but regen happens 3 days later, it is over-estimating.

### 4.3 Business Metrics for Success
*   **Reduction in Unplanned Downtime**: Measuring the decrease in roadside breakdowns due to DPF clogs.
*   **Maintenance ROI**: Cost of preventive regen vs. cost of failure.
*   **False Alarm Rate**: Tracking "Driver Complaints" about unnecessary regen warnings to ensure trust.