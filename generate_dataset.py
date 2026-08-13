import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n_samples = 1000

# Synthetic medical dataset generation based on clinical triage guidelines
data = {
    'patient_id': range(1, n_samples + 1),
    'age': np.random.randint(1, 95, n_samples),
    'heart_rate': np.random.randint(45, 160, n_samples),       # Normal: 60-100
    'spo2': np.random.randint(75, 100, n_samples),            # Normal: 95-100
    'temperature': np.round(np.random.uniform(96.0, 105.0, n_samples), 1), # Normal: 98.6
    'systolic_bp': np.random.randint(80, 190, n_samples),     # Blood Pressure
    'diastolic_bp': np.random.randint(50, 120, n_samples),
}

df = pd.DataFrame(data)

# Clinical Triage Labeling Rules (0 = Low, 1 = Medium, 2 = High, 3 = Critical)
conditions = [
    (df['spo2'] < 88) | (df['heart_rate'] > 135) | (df['systolic_bp'] > 170),  # Critical
    (df['spo2'] < 92) | (df['heart_rate'] > 115) | (df['temperature'] > 102.5), # High
    (df['spo2'] < 95) | (df['heart_rate'] > 100) | (df['temperature'] > 100.4)  # Medium
]
choices = [3, 2, 1]
df['priority'] = np.select(conditions, choices, default=0) # Default is Low

# Save to CSV
df.to_csv("data/patient_history.csv", index=False)
print("✅ Dataset created successfully at 'data/patient_history.csv' with 1000 records!")