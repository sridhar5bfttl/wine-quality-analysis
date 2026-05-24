import os
import pickle
import pandas as pd
import numpy as np

def run_verification():
    print("=== PROGRAMMATIC VERIFICATION OF WINE WORKSPACE ===")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    
    # 1. Verify datasets
    print("\n1. Verifying Datasets...")
    red_data_path = os.path.join(data_dir, "winequality-red.csv")
    white_data_path = os.path.join(data_dir, "winequality-white.csv")
    
    assert os.path.exists(red_data_path), "Red wine CSV missing!"
    assert os.path.exists(white_data_path), "White wine CSV missing!"
    
    red_df = pd.read_csv(red_data_path, sep=';')
    white_df = pd.read_csv(white_data_path, sep=';')
    
    print(f"  Red wine dataset: {red_df.shape[0]} rows, {red_df.shape[1]} columns")
    print(f"  White wine dataset: {white_df.shape[0]} rows, {white_df.shape[1]} columns")
    
    expected_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality'
    ]
    for col in expected_cols:
        assert col in red_df.columns, f"Column '{col}' missing in red wine dataset!"
        assert col in white_df.columns, f"Column '{col}' missing in white wine dataset!"
    print("  Dataset schemas: OK")
    
    # 2. Verify models
    print("\n2. Verifying Models and Feature Importances...")
    red_model_path = os.path.join(data_dir, "red_wine_model.pkl")
    white_model_path = os.path.join(data_dir, "white_wine_model.pkl")
    
    assert os.path.exists(red_model_path), "Red wine model PKL missing!"
    assert os.path.exists(white_model_path), "White wine model PKL missing!"
    
    with open(red_model_path, 'rb') as f:
        red_model_data = pickle.load(f)
    with open(white_model_path, 'rb') as f:
        white_model_data = pickle.load(f)
        
    print("  Models loaded successfully.")
    
    # Assert model keys
    for key in ['model', 'features', 'wine_type', 'feature_importances']:
        assert key in red_model_data, f"Key '{key}' missing in red model pickle!"
        assert key in white_model_data, f"Key '{key}' missing in white model pickle!"
        
    print(f"  Red model type: {type(red_model_data['model'])}")
    print(f"  White model type: {type(white_model_data['model'])}")
    
    # 3. Verify Prediction Logic
    print("\n3. Testing Real-time Inference Simulation...")
    # Test red wine prediction with mean values
    red_inputs = pd.DataFrame([{
        'fixed acidity': 8.32,
        'volatile acidity': 0.53,
        'citric acid': 0.27,
        'residual sugar': 2.54,
        'chlorides': 0.09,
        'free sulfur dioxide': 15.87,
        'total sulfur dioxide': 46.47,
        'density': 0.9967,
        'pH': 3.31,
        'sulphates': 0.66,
        'alcohol': 10.42
    }])
    red_inputs = red_inputs[red_model_data['features']] # align
    red_pred = red_model_data['model'].predict(red_inputs)[0]
    red_probs = red_model_data['model'].predict_proba(red_inputs)[0]
    print(f"  Red Wine Mean Input Prediction: Class {red_pred} (Prob Not Good: {red_probs[0]:.2f}, Prob Good: {red_probs[1]:.2f})")
    assert 0.0 <= red_probs[0] <= 1.0, "Probability out of bounds!"
    
    # Test white wine prediction with mean values
    white_inputs = pd.DataFrame([{
        'fixed acidity': 6.85,
        'volatile acidity': 0.28,
        'citric acid': 0.32,
        'residual sugar': 6.39,
        'chlorides': 0.046,
        'free sulfur dioxide': 35.31,
        'total sulfur dioxide': 138.36,
        'density': 0.9940,
        'pH': 3.19,
        'sulphates': 0.49,
        'alcohol': 10.51
    }])
    white_inputs = white_inputs[white_model_data['features']] # align
    white_pred = white_model_data['model'].predict(white_inputs)[0]
    white_probs = white_model_data['model'].predict_proba(white_inputs)[0]
    print(f"  White Wine Mean Input Prediction: Class {white_pred} (Prob Not Good: {white_probs[0]:.2f}, Prob Good: {white_probs[1]:.2f})")
    assert 0.0 <= white_probs[0] <= 1.0, "Probability out of bounds!"
    
    print("\nVERIFICATION STATUS: ALL TESTS PASSED SUCCESSFULLY! 🍷")

if __name__ == "__main__":
    run_verification()
