import pandas as pd
from pybaseball import statcast

def fetch_statcast_sample():
    print("Initializing MLB Data Pipeline...")
    print("Fetching raw Statcast data. This usually takes 30-60 seconds...")
    raw_data = statcast(start_dt='2024-05-01', end_dt='2024-05-07')
    print(f"\nSuccess! Downloaded {len(raw_data)} pitches.")
    
    kinematic_columns = [
        'pitch_type', 'release_speed', 'release_spin_rate', 
        'events', 'description', 'launch_speed', 'launch_angle',
        'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle'
    ]
    return raw_data[kinematic_columns]

def transform_data(df):
    print("\nStarting Feature Engineering (Transformation)...")
    
    # 1. Filter out pitches with no contact. 
    # If there is no exit velocity (launch_speed), it wasn't a Batted Ball Event (BBE).
    bbe_df = df.dropna(subset=['launch_speed', 'launch_angle']).copy()
    print(f"Filtered down to {len(bbe_df)} Batted Ball Events.")
    
    # 2. Calculate Hard Hit (Exit Velocity >= 95 mph)
    # The .astype(int) turns the True/False result into a 1 or 0 for our ML model.
    bbe_df['is_hard_hit'] = (bbe_df['launch_speed'] >= 95.0).astype(int)
    
    # 3. Calculate Barrel (Proxy: EV >= 98 & Launch Angle 26-30 degrees)
    bbe_df['is_barrel'] = ((bbe_df['launch_speed'] >= 98.0) & 
                           (bbe_df['launch_angle'] >= 26.0) & 
                           (bbe_df['launch_angle'] <= 30.0)).astype(int)
                           
    # 4. Calculate Blast / BBomb (Proxy: Elite EV >= 105 & optimal HR angle)
    bbe_df['is_blast'] = ((bbe_df['launch_speed'] >= 105.0) & 
                          (bbe_df['launch_angle'] >= 20.0) & 
                          (bbe_df['launch_angle'] <= 35.0)).astype(int)
    
    print("\nTransformation Complete! Here are our new ML Features:")
    # We display our new columns to verify the math worked
    print(bbe_df[['pitch_type', 'launch_speed', 'launch_angle', 'is_hard_hit', 'is_barrel', 'is_blast']].head(15))
    
    return bbe_df

# Execute the pipeline
if __name__ == "__main__":
    raw_df = fetch_statcast_sample()
    clean_ml_data = transform_data(raw_df)