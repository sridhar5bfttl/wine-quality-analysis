import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_virtual_flow_meter(data_path, model_path):
    print("===============================================")
    print("⚙️ Training Virtual Flow Metering Regression Model")
    print("===============================================")
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Filter out days where the well was shut down (on_stream_hours == 0)
    # because predicting flow during shutdown is trivial (it's 0)
    df_active = df[df["on_stream_hours"] > 0].copy()
    
    # Define features (operational sensor readings) and target (oil volume)
    features = [
        "on_stream_hours",
        "choke_size_percentage",
        "bottomhole_pressure_psi",
        "bottomhole_temperature_f",
        "wellhead_pressure_psi",
        "wellhead_temperature_f"
    ]
    target = "oil_volume_bbl"
    
    X = df_active[features]
    y = df_active[target]
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Dataset Size (Active Days): {len(df_active)}")
    print(f"R² Score:                   {r2:.4f}")
    print(f"Mean Absolute Error (MAE):  {mae:.2f} bbl/day")
    print(f"Root Mean Squared Error:    {rmse:.2f} bbl/day")
    print("-----------------------------------------------")
    
    # Feature Importances
    importances = model.feature_importances_
    for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"Feature '{name}': {imp*100:.2f}% importance")
    
    # Save the model and feature metadata
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    payload = {
        "model": model,
        "features": features,
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse}
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"\n✅ Model successfully saved to {model_path}")
    print("===============================================")

if __name__ == "__main__":
    train_virtual_flow_meter(
        data_path="oil_gas_production/data/well_production_data.csv",
        model_path="oil_gas_production/data/virtual_flow_meter_model.pkl"
    )
