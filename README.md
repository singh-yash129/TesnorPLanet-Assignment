DPF Predictive Maintenance Pipeline

This project implements an End-to-End Machine Learning pipeline to predict soot load in Diesel Particulate Filters (DPF). It includes data generation, training, and a serving API.

Project Structure

data_generator.py: Generates synthetic vehicle telemetry (Physics-based simulation).

train_model.py: Ingests data, creates features, trains XGBoost, saves model.

app.py: FastAPI application for real-time predictions.

test_suite.py: Unit and Integration tests.

Dockerfile: Container configuration.

Setup & Running

1. Install Dependencies

pip install -r requirements.txt


2. Generate Data

This will create data/raw/ with telemetry, maintenance, and trip CSVs.

python data_generator.py


3. Train Model

Trains the XGBoost regressor and saves dpf_model_v1.pkl to models/.

python train_model.py


4. Run API Server

Starts the local server at http://0.0.0.0:8000.

uvicorn app:app --reload


5. Test the API

You can access the auto-generated Swagger docs at: http://127.0.0.1:8000/docs

Or run the test suite:

pytest test_suite.py


Docker Build

To build and run the containerized application:

docker build -t dpf-predictor .
docker run -p 8000:8000 dpf-predictor
