DPF Soot Load Prediction: Technical Report

Author: Candidate for Tensor Planet Internship

Date: October 2023

1. Data Generation & Engineering Approach

1.1 Synthetic Data Strategy

Since real DPF data wasn't provided, I built a physics-informed generator (data_generator.py).

The Physics Model: Soot doesn't accumulate linearly. It accumulates based on engine_load (dirty combustion) and reduces based on exhaust_temp (passive regeneration > 300°C).

The Sensor Proxy: The primary indicator of soot is differential_pressure. However, this sensor is noisy and confounded by exhaust_flow. A clean filter at high RPM reads higher pressure than a dirty filter at idle. My generator simulates this confounding variable ($P = R \times Q$), forcing the ML model to learn the relationship, not just memorize the pressure value.

1.2 Feature Engineering Choices

Raw telemetry is noisy. I focused on features that isolate the filter's resistance from the engine's operation:

Rolling Means (15m, 1h): Smooths out sensor spikes caused by bumps or sudden acceleration.

Pressure/Flow Ratio: Approximates the physical resistance of the filter, which is a cleaner proxy for soot mass than raw pressure.

Differential Pressure Delta: Captures the rate of clogging.

2. Modeling Rationale

2.1 Problem Framing

I framed this as a Regression Problem (Target: Soot Load in grams).

Why not classification? A binary "Regen Needed" label loses information. Predicting the actual load allows the fleet manager to set dynamic thresholds (e.g., "Regen at 80% if near a service center, else push to 90%").

2.2 Model Selection: XGBoost

I chose XGBoost (Gradient Boosted Trees) over Linear Regression or Deep Learning (LSTM).

Non-Linearity: The relationship between RPM, Temp, and Pressure is non-linear.

Robustness: Tree models handle outliers and unscaled data better than Neural Networks, making them safer for industrial sensor data.

Interpretability: Feature importance scores help engineers diagnose which sensors are driving predictions.

2.3 Business Tradeoffs (Precision vs. Recall)

In Predictive Maintenance, the costs are asymmetric:

False Negative (Missed Breakdown): Cost is high (Towing, late delivery, engine damage).

False Positive (Early Regen): Cost is low (Small fuel penalty).

Strategy: The model should optimize for Recall. In the API, I implemented a tiered warning system. A "Warning" is issued at 75% load, providing a safety buffer before the 90% "Critical" level.

3. Production & MLOps Architecture

3.1 API Design

The solution is served via FastAPI. It handles:

Single Inference: For real-time dashboard updates.

Batch Inference: For end-of-day fleet analysis.

Cold Start: The feature extractor handles missing history (first data point of the day) by using instantaneous values, preventing API crashes on new vehicles.

3.2 Robustness & Quality Assurance

Input Validation: Pydantic ensures data types match.

Zero-Division Protection: The code explicitly handles cases where exhaust_flow is zero (engine off) to prevent mathematical errors.

Unit Tests: Included in test_suite.py to verify feature calculation logic before deployment.

4. Recommendations

Deployment: Deploy the Docker container to a Kubernetes cluster. Use a sidecar pattern to fetch model artifacts from S3.

Model Improvement: Integrate "Time Since Last Regen" as a feature. This requires joining the Maintenance table with Telemetry in a production feature store.

Feedback Loop: Capture actual regeneration events to retrain the model. If the model says 90% soot, but the regen happens 3 days later, the model is over-estimating.