import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "models/triage_xgboost.pkl"
DATA_PATH = "data/patient_history.csv"

def generate_and_train():
    """Generates a dataset, performs train/test split, and trains the model."""
    print("--- 1. Generating Medical Dataset ---")
    np.random.seed(42)
    n_samples = 1000

    data = {
        'age': np.random.randint(1, 95, n_samples),
        'heart_rate': np.random.randint(45, 160, n_samples),
        'spo2': np.random.randint(75, 100, n_samples),
        'temperature': np.round(np.random.uniform(96.0, 105.0, n_samples), 1),
        'systolic_bp': np.random.randint(80, 190, n_samples),
        'diastolic_bp': np.random.randint(50, 120, n_samples),
    }
    df = pd.DataFrame(data)

    # Triage Rules (0 = Low, 1 = Medium, 2 = High, 3 = Critical)
    conditions = [
        (df['spo2'] < 88) | (df['heart_rate'] > 135) | (df['systolic_bp'] > 170),
        (df['spo2'] < 92) | (df['heart_rate'] > 115) | (df['temperature'] > 102.5),
        (df['spo2'] < 95) | (df['heart_rate'] > 100) | (df['temperature'] > 100.4)
    ]
    df['priority'] = np.select(conditions, [3, 2, 1], default=0)

    # Save dataset so you have it for your presentation
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}")

    print("\n--- 2. Train / Test Split ---")
    X = df[['age', 'heart_rate', 'spo2', 'temperature', 'systolic_bp', 'diastolic_bp']]
    y = df['priority']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print("\n--- 3. Training XGBoost Model ---")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("\n--- 4. Evaluating Model ---")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Critical"]))

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {MODEL_PATH}")

def predict_triage(age, hr, spo2, temp, sys_bp=120, dia_bp=80):
    """Loads the model and predicts priority for a new patient from the web app."""
    
    # Auto-train if the model doesn't exist yet
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Initializing training pipeline...")
        generate_and_train()
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    # Create a dataframe for the single patient matching the training features
    input_df = pd.DataFrame([{
        'age': age,
        'heart_rate': hr,
        'spo2': spo2,
        'temperature': temp,
        'systolic_bp': sys_bp,
        'diastolic_bp': dia_bp
    }])
    
    probs = model.predict_proba(input_df)[0]
    prediction = model.predict(input_df)[0]
    
    priority_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    
    # Calculate a severity score between 0.0 and 1.0 based on probabilities
    severity_score = float(probs[3] * 1.0 + probs[2] * 0.7 + probs[1] * 0.3)
    
    return priority_map[prediction], round(severity_score, 2)

if __name__ == "__main__":
    # If you run this file directly (python ml_engine.py), it will train the model.
    generate_and_train()