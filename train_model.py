import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DPFDataPipeline:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
    
    def load_data(self):
        print("Loading data...")
        try:
            df = pd.read_csv('data/raw/telemetry.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['vehicle_id', 'timestamp'])
            return df
        except FileNotFoundError:
            raise Exception("Data not found. Run data_generator.py first.")

    def feature_engineering(self, df):
        print("Engineering features...")
        
        # 1. Rolling Statistics (Capture trends vs instantaneous noise)
        # We group by vehicle to ensure stats don't bleed between trucks
        windows = [3, 12] # 15 mins, 1 hour (assuming 5 min intervals)
        
        for w in windows:
            df[f'exhaust_temp_rolling_mean_{w}'] = df.groupby('vehicle_id')['exhaust_temp_c'].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f'diff_pressure_rolling_mean_{w}'] = df.groupby('vehicle_id')['diff_pressure_kpa'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        
        # 2. Physics-based ratios
        # High pressure with low flow suggests clogging
        # We add 1e-5 to avoid division by zero
        df['pressure_flow_ratio'] = df['diff_pressure_kpa'] / (df['exhaust_flow_rate_kg_h'] + 1e-5)
        
        # 3. Temporal Features (Time since last regen is critical, but we'll proxy it 
        # with a simpler cumulative sum reset logic if we had maintenance logs joined. 
        # For this assignment, we use cumulative sums of load as a proxy for soot buildup potential)
        df['cumulative_load_proxy'] = df.groupby('vehicle_id')['engine_load_pct'].cumsum()
        
        # 4. Lag features (Change in pressure)
        df['diff_pressure_delta'] = df.groupby('vehicle_id')['diff_pressure_kpa'].diff().fillna(0)
        
        # Define features for modeling
        self.feature_cols = [
            'engine_rpm', 'engine_load_pct', 'exhaust_temp_c', 
            'diff_pressure_kpa', 'exhaust_flow_rate_kg_h',
            'exhaust_temp_rolling_mean_12', 'diff_pressure_rolling_mean_12',
            'pressure_flow_ratio', 'diff_pressure_delta'
        ]
        
        # Clean NaNs created by rolling
        df = df.dropna()
        
        return df

    def train(self, df):
        print("Training model...")
        X = df[self.feature_cols]
        y = df['soot_load_ground_truth']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost Regressor
        # Objective: Minimize squared error.
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        print(f"\nModel Evaluation:\nMAE: {mae:.4f} grams\nRMSE: {rmse:.4f} grams")
        print(f"Mean Target Value: {y.mean():.4f}")
        
        return {"mae": mae, "rmse": rmse}

    def save_artifacts(self, version="v1"):
        print(f"Saving artifacts version {version}...")
        os.makedirs('models', exist_ok=True)
        
        artifacts = {
            "model": self.model,
            "features": self.feature_cols,
            "version": version,
            "timestamp": pd.Timestamp.now()
        }
        
        joblib.dump(artifacts, f'models/dpf_model_{version}.pkl')
        print("Done.")

if __name__ == "__main__":
    pipeline = DPFDataPipeline()
    raw_data = pipeline.load_data()
    processed_data = pipeline.feature_engineering(raw_data)
    metrics = pipeline.train(processed_data)
    pipeline.save_artifacts()