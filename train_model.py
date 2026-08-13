import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

def train_and_evaluate():
    print("--- 1. Loading Dataset ---")
    dataset_path = "data/patient_history.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found! Run generate_dataset.py first.")
        
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} patient records.")

    # 2. Feature Selection & Target Variable
    X = df[['age', 'heart_rate', 'spo2', 'temperature', 'systolic_bp', 'diastolic_bp']]
    y = df['priority']

    # 3. Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n--- 2. Train-Test Split ---")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples:  {X_test.shape[0]}")

    # 4. Model Training using XGBoost
    print("\n--- 3. Training XGBoost Model ---")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    # 5. Model Testing & Evaluation
    print("\n--- 4. Evaluating Model on Unseen Test Data ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%\n")
    
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Critical"]))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Save Trained Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/triage_xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model trained and saved to '{model_path}'!")

if __name__ == "__main__":
    train_and_evaluate()