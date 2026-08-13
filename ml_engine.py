import pandas as pd
import xgboost as xgb
import pickle
import os
import numpy as np

MODEL_PATH = "models/triage_xgboost.pkl"

def train_dummy_model():
    """Trains a quick XGBoost model on synthetic data to ensure the project runs immediately."""
    
    # Generate 500 rows of synthetic patient data
    data = {
        'age': np.random.randint(18, 90, 500),
        'hr': np.random.randint(50, 150, 500),         # Heart Rate
        'spo2': np.random.randint(80, 100, 500),       # Oxygen saturation
        'temp': np.random.uniform(97.0, 104.0, 500),   # Temperature
        'severity_score': np.random.uniform(0.1, 1.0, 500)
    }
    
    # Assign priorities based on critical thresholds
    # 0 = Low, 1 = Medium, 2 = High, 3 = Critical
    conditions = [
        (data['severity_score'] > 0.8) | (data['spo2'] < 88),
        (data['severity_score'] > 0.6) | (data['hr'] > 120),
        (data['severity_score'] > 0.3)
    ]
    choices = [3, 2, 1]
    data['priority'] = np.select(conditions, choices, default=0)
    
    df = pd.DataFrame(data)
    
    X = df[['age', 'hr', 'spo2', 'temp']]
    y = df['priority']
    
    # Train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    
    # Save the model to the models directory
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model successfully trained and saved to {MODEL_PATH}")

def predict_triage(age, hr, spo2, temp):
    """Loads the model and predicts the priority level for a new patient."""
    
    # Auto-train if the model doesn't exist yet
    if not os.path.exists(MODEL_PATH):
        train_dummy_model()
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    input_data = pd.DataFrame({'age': [age], 'hr': [hr], 'spo2': [spo2], 'temp': [temp]})
    
    probs = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]
    
    priority_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    
    # Calculate a custom severity score between 0.0 and 1.0 based on probabilities
    severity_score = float(probs[3] + (probs[2] * 0.5))
    
    return priority_map[prediction], round(severity_score, 2)

if __name__ == "__main__":
    train_dummy_model()