import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(n_vehicles=10, days=30):
    """
    Generates synthetic telemetry, maintenance, and trip data for DPF monitoring.
    """
    np.random.seed(42)
    
    telemetry_data = []
    maintenance_data = []
    trip_data = []
    
    print(f"Generating data for {n_vehicles} vehicles over {days} days...")

    for vehicle_id in range(1001, 1001 + n_vehicles):
        current_time = datetime.now() - timedelta(days=days)
        end_time = datetime.now()
        
        # Initial State
        soot_load_grams = np.random.uniform(0, 10) 
        cumulative_dist = 0
        
        while current_time < end_time:
            # Create a Trip
            trip_duration_hours = np.random.uniform(1, 8)
            trip_end = current_time + timedelta(hours=trip_duration_hours)
            trip_dist = np.random.uniform(30, 80) * trip_duration_hours
            
            # Trip Characteristics
            driving_mode = np.random.choice(['highway', 'city', 'idle', 'mixed'], p=[0.4, 0.3, 0.1, 0.2])
            trip_data.append({
                'vehicle_id': vehicle_id,
                'trip_id': f"T_{vehicle_id}_{int(current_time.timestamp())}",
                'start_time': current_time,
                'end_time': trip_end,
                'distance_km': trip_dist,
                'driving_mode': driving_mode
            })
            
            # Generate Telemetry Points (every 5 minutes)
            temp_time = current_time
            while temp_time < trip_end:
                # 1. Simulate Engine Parameters based on mode
                if driving_mode == 'highway':
                    rpm = np.random.normal(1500, 100)
                    load = np.random.normal(70, 10)
                    speed = np.random.normal(85, 5)
                    exhaust_temp = np.random.normal(350, 30) # Hotter
                elif driving_mode == 'city':
                    rpm = np.random.normal(1100, 200)
                    load = np.random.normal(40, 15)
                    speed = np.random.normal(30, 15)
                    exhaust_temp = np.random.normal(250, 40) # Cooler
                else: # idle/mixed
                    rpm = np.random.normal(700, 50)
                    load = np.random.normal(10, 5)
                    speed = 0
                    exhaust_temp = np.random.normal(180, 20)

                # 2. Physics Simulation: Soot Accumulation vs Burn (Passive Regen)
                # Soot builds up with load, burns off if temp > 300C
                accumulation_rate = (load / 100) * 0.5  # grams per 5 mins
                burn_rate = 0
                if exhaust_temp > 320:
                    burn_rate = ((exhaust_temp - 300) / 100) * 0.8
                
                soot_load_grams = max(0, soot_load_grams + accumulation_rate - burn_rate)
                
                # 3. Sensor Readings (Noisy)
                # Differential Pressure (kPa) is function of Soot Load AND Exhaust Flow (proxy by RPM/Load)
                flow_rate_proxy = (rpm * load) / 50000
                diff_pressure = (soot_load_grams * 0.1 * flow_rate_proxy) + np.random.normal(0, 0.05)
                
                # 4. Maintenance Event Logic (Active Regeneration)
                # If soot gets too high, force a regen event
                if soot_load_grams > 45:
                    maintenance_data.append({
                        'vehicle_id': vehicle_id,
                        'timestamp': temp_time,
                        'type': 'active_regeneration',
                        'trigger': 'soot_threshold_exceeded'
                    })
                    soot_load_grams = 0 # Reset
                    # Add a marker in telemetry that regen happened
                    active_regen_status = 1
                else:
                    active_regen_status = 0

                telemetry_data.append({
                    'vehicle_id': vehicle_id,
                    'timestamp': temp_time,
                    'engine_rpm': int(rpm),
                    'engine_load_pct': round(load, 2),
                    'exhaust_temp_c': round(exhaust_temp, 1),
                    'exhaust_flow_rate_kg_h': round(flow_rate_proxy * 100, 2), # Synthetic proxy
                    'diff_pressure_kpa': round(max(0, diff_pressure), 3),
                    'vehicle_speed_kmh': round(speed, 1),
                    'ambient_temp_c': round(np.random.normal(25, 5), 1),
                    'active_regen_status': active_regen_status,
                    'soot_load_ground_truth': round(soot_load_grams, 2) # TARGET VARIABLE
                })
                
                temp_time += timedelta(minutes=5)
            
            # Rest period between trips
            current_time = trip_end + timedelta(hours=np.random.uniform(2, 12))
            cumulative_dist += trip_dist

    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    
    df_tel = pd.DataFrame(telemetry_data)
    df_tel.to_csv('data/raw/telemetry.csv', index=False)
    
    df_maint = pd.DataFrame(maintenance_data)
    df_maint.to_csv('data/raw/maintenance.csv', index=False)
    
    df_trip = pd.DataFrame(trip_data)
    df_trip.to_csv('data/raw/trips.csv', index=False)
    
    print("Data generation complete. Saved to data/raw/")

if __name__ == "__main__":
    generate_synthetic_data()