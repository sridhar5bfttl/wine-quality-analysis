import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_wine_model(wine_type, data_path, models_dir):
    print(f"\n--- Training Model for {wine_type.upper()} Wine ---")
    
    # Load dataset (separated by semicolon)
    df = pd.read_csv(data_path, sep=';')
    print(f"Loaded {len(df)} samples from {os.path.basename(data_path)}")
    
    # Define features and target
    # Target is binary: quality >= 6 is considered "Good" (1), else "Not Good" (0)
    X = df.drop(columns=['quality'])
    y = (df['quality'] >= 6).astype(int)
    
    print(f"Class distribution:")
    print(f"  Good (quality >= 6): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  Not Good (quality < 6): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Good", "Good"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nTop 5 Important Features:")
    print(importance.head(5).to_string(index=False))
    
    # Save the model and feature columns list
    model_data = {
        'model': clf,
        'features': list(X.columns),
        'wine_type': wine_type,
        'feature_importances': importance.to_dict(orient='records')
    }
    
    model_path = os.path.join(models_dir, f"{wine_type}_wine_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved model to {model_path}")

def main():
    # Set paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    
    red_data_path = os.path.join(data_dir, "winequality-red.csv")
    white_data_path = os.path.join(data_dir, "winequality-white.csv")
    
    train_wine_model("red", red_data_path, data_dir)
    train_wine_model("white", white_data_path, data_dir)

if __name__ == "__main__":
    main()
