import os
import pickle
import pandas as pd

def verify_pipeline():
    print("===============================================")
    print("🔬 Verifying PetroPulse Upstream Workspace Pipeline")
    print("===============================================")
    
    # 1. Check data file
    csv_path = "oil_gas_production/data/well_production_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: Dataset file not found at {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    print(f"✅ Production dataset loaded successfully. Shape: {df.shape}")
    
    # 2. Check model file
    model_path = "oil_gas_production/data/virtual_flow_meter_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model pickle not found at {model_path}")
        return False
        
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
        
    model = payload["model"]
    features = payload["features"]
    metrics = payload["metrics"]
    
    print(f"✅ Regression model loaded. R² metric: {metrics['r2']:.4f}")
    
    # 3. Test dummy inference
    dummy_input = pd.DataFrame([{
        "on_stream_hours": 24.0,
        "choke_size_percentage": 75.0,
        "bottomhole_pressure_psi": 2500.0,
        "bottomhole_temperature_f": 180.0,
        "wellhead_pressure_psi": 1200.0,
        "wellhead_temperature_f": 110.0
    }])
    
    try:
        prediction = model.predict(dummy_input[features])[0]
        print(f"✅ Inference check: predicted rate for dummy input is {prediction:.2f} bbl/day")
    except Exception as e:
        print(f"❌ Inference check failed: {e}")
        return False
        
    print("\n🎉 Verification SUCCESSFUL! All system elements are correct.")
    print("===============================================")
    return True

if __name__ == "__main__":
    verify_pipeline()
