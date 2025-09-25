import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ----------------- AQI Breakpoints (US-EPA) -----------------
aqi_breakpoints = {
    "PM2.5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "PM10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "NO2": [
        (0.0, 53.0, 0, 50),
        (54.0, 100.0, 51, 100),
        (101.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300),
        (1250.0, 1649.0, 301, 400),
        (1650.0, 2049.0, 401, 500),
    ],
    "SO2": [
        (0.0, 35.0, 0, 50),
        (36.0, 75.0, 51, 100),
        (76.0, 185.0, 101, 150),
        (186.0, 304.0, 151, 200),
        (305.0, 604.0, 201, 300),
        (605.0, 804.0, 301, 400),
        (805.0, 1004.0, 401, 500),
    ],
    "CO": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
    "OZONE": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ]
}

# ----------------- AQI Calculation Functions -----------------
def calculate_individual_aqi(concentration, pollutant):
    """Calculate AQI for a single pollutant concentration using US-EPA breakpoints."""
    if pollutant not in aqi_breakpoints or pd.isna(concentration):
        return None
    for (C_low, C_high, I_low, I_high) in aqi_breakpoints[pollutant]:
        if C_low <= concentration <= C_high:
            return round(((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low)
    return None  # out of range


def calculate_overall_aqi(row, pollutants):
    """Calculate overall AQI = max of sub-indices."""
    sub_indices = []
    for p in pollutants:
        val = row[p]
        aqi_val = calculate_individual_aqi(val, p)
        if aqi_val is not None:
            sub_indices.append(aqi_val)
    return max(sub_indices) if sub_indices else None


# ----------------- Preprocessing Function -----------------
def preprocess_data(filepath, selected_pollutants=None, target='AQI'):
    """
    Preprocess AQI dataset from long format to wide format and prepare for ML.
    """
    # 1. Load dataset
    df = pd.read_csv(filepath)

    # 2. Convert date
    df['last_update'] = pd.to_datetime(df['last_update'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df['year'] = df['last_update'].dt.year
    df['month'] = df['last_update'].dt.month
    df['hour'] = df['last_update'].dt.hour

    # 3. Drop rows with missing pollutant_avg
    df = df.dropna(subset=['pollutant_avg'])

    # 4. Pivot table: each pollutant as a column
    df_wide = df.pivot_table(
        index=['country', 'state', 'city', 'station', 'last_update', 'year', 'month', 'hour', 'latitude', 'longitude'],
        columns='pollutant_id',
        values='pollutant_avg',
        aggfunc='mean'
    ).reset_index()

    # 5. Remove column names from pivot
    df_wide.columns.name = None

    # 6. Drop rows with NaN after pivot
    df_wide = df_wide.dropna()

    # 7. Define pollutants
    if selected_pollutants is None:
        selected_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']

    # 8. Calculate AQI using official formula
    df_wide['AQI'] = df_wide.apply(lambda row: calculate_overall_aqi(row, selected_pollutants), axis=1)

    # Drop rows where AQI couldn't be calculated
    df_wide = df_wide.dropna(subset=['AQI'])

    # 9. Split X, y
    X = df_wide[selected_pollutants]
    y = df_wide['AQI']

    # 10. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 11. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 12. Save processed file in the same folder as the original dataset
    processed_file_path = os.path.join(
        os.path.dirname(filepath),
        "processed_AQI_US_EPA.csv"
    )
    df_wide.to_csv(processed_file_path, index=False)
    print(f"✅ Processed file is saved to {processed_file_path}")
    print(f"Shape of processed data: {df_wide.shape}")

    return X_train, X_test, y_train, y_test, scaler, df_wide


# ----------------- Run Preprocessing -----------------
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, scaler, df_clean = preprocess_data(
            r'D:\WORK\air_quality_project\data\raw\AQI.csv',
            selected_pollutants=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your file path and data structure")
