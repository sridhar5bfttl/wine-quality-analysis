import os
import urllib.request
import numpy as np
import pandas as pd

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Successfully downloaded to {dest_path}")

def generate_synthetic_wine_data(wine_type, dest_path):
    print(f"Generating synthetic {wine_type} wine quality dataset at {dest_path}...")
    np.random.seed(42 if wine_type == "red" else 43)
    
    if wine_type == "red":
        n_samples = 1599
        # Mean and std for features
        stats = {
            'fixed acidity': (8.32, 1.74, 4.6, 15.9),
            'volatile acidity': (0.53, 0.18, 0.12, 1.58),
            'citric acid': (0.27, 0.19, 0.0, 1.0),
            'residual sugar': (2.54, 1.41, 0.9, 15.5),
            'chlorides': (0.09, 0.05, 0.01, 0.61),
            'free sulfur dioxide': (15.87, 10.46, 1.0, 72.0),
            'total sulfur dioxide': (46.47, 32.95, 6.0, 289.0),
            'density': (0.9967, 0.0019, 0.990, 1.004),
            'pH': (3.31, 0.15, 2.74, 4.01),
            'sulphates': (0.66, 0.17, 0.33, 2.0),
            'alcohol': (10.42, 1.07, 8.4, 14.9)
        }
    else:
        n_samples = 4898
        stats = {
            'fixed acidity': (6.85, 0.84, 3.8, 14.2),
            'volatile acidity': (0.28, 0.10, 0.08, 1.10),
            'citric acid': (0.32, 0.12, 0.0, 1.66),
            'residual sugar': (6.39, 5.07, 0.6, 65.8),
            'chlorides': (0.046, 0.022, 0.009, 0.346),
            'free sulfur dioxide': (35.31, 17.01, 2.0, 289.0),
            'total sulfur dioxide': (138.36, 42.50, 9.0, 440.0),
            'density': (0.9940, 0.0030, 0.987, 1.039),
            'pH': (3.19, 0.15, 2.72, 3.82),
            'sulphates': (0.49, 0.11, 0.22, 1.08),
            'alcohol': (10.51, 1.23, 8.0, 14.2)
        }

    # Generate correlated features
    df = pd.DataFrame()
    for col, (mean, std, min_val, max_val) in stats.items():
        # Generate log-normal for right-skewed features, normal for others
        if col in ['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']:
            # Adjust log-normal parameters to match target mean and std
            s = np.sqrt(np.log(1 + (std / mean)**2))
            mu = np.log(mean) - 0.5 * s**2
            vals = np.random.lognormal(mean=mu, sigma=s, size=n_samples)
        else:
            vals = np.random.normal(loc=mean, scale=std, size=n_samples)
            
        df[col] = np.clip(vals, min_val, max_val)

    # Let's create a realistic quality target that depends on the features + noise
    # Standard Wine quality correlations:
    # + alcohol, + sulphates, + citric acid, - volatile acidity, - density, - chlorides
    if wine_type == "red":
        score = (
            0.4 * (df['alcohol'] - 10.42) / 1.07
            + 0.2 * (df['sulphates'] - 0.66) / 0.17
            + 0.1 * (df['citric acid'] - 0.27) / 0.19
            - 0.3 * (df['volatile acidity'] - 0.53) / 0.18
            - 0.15 * (df['density'] - 0.9967) / 0.0019
            - 0.1 * (df['chlorides'] - 0.09) / 0.05
            - 0.05 * (df['total sulfur dioxide'] - 46.47) / 32.95
            + np.random.normal(0, 0.55, size=n_samples)
        )
        # Map score to integer quality 3-8 (mean 5.64, std 0.81)
        quality = 5.64 + score * 0.81
        quality_int = np.clip(np.round(quality), 3, 8).astype(int)
    else:
        score = (
            0.45 * (df['alcohol'] - 10.51) / 1.23
            + 0.1 * (df['sulphates'] - 0.49) / 0.11
            - 0.25 * (df['volatile acidity'] - 0.28) / 0.10
            - 0.25 * (df['density'] - 0.9940) / 0.0030
            - 0.1 * (df['chlorides'] - 0.046) / 0.022
            + 0.05 * (df['free sulfur dioxide'] - 35.31) / 17.01
            + np.random.normal(0, 0.5, size=n_samples)
        )
        # Map score to integer quality 3-9 (mean 5.88, std 0.89)
        quality = 5.88 + score * 0.89
        quality_int = np.clip(np.round(quality), 3, 9).astype(int)

    df['quality'] = quality_int
    
    # Save as semicolon separated file, as in original UCI dataset
    df.to_csv(dest_path, sep=';', index=False)
    print(f"Successfully generated dataset with {len(df)} rows.")

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    red_dest = os.path.join(data_dir, "winequality-red.csv")
    white_dest = os.path.join(data_dir, "winequality-white.csv")
    
    # Try downloading red wine
    try:
        download_file(red_wine_url, red_dest)
    except Exception as e:
        print(f"Network download failed: {e}")
        generate_synthetic_wine_data("red", red_dest)
        
    # Try downloading white wine
    try:
        download_file(white_wine_url, white_dest)
    except Exception as e:
        print(f"Network download failed: {e}")
        generate_synthetic_wine_data("white", white_dest)

if __name__ == "__main__":
    main()
