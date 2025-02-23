import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('/kaggle/input/predictive-maintenance-dataset-ai4i-2020/ai4i2020.csv')

# Rename columns for consistency
df.rename(columns={
    'UDI': 'UID',
    'Process temperature [K]': 'Process_Temperature_K',
    'Air temperature [K]': 'Air_Temperature_K',
    'Rotational speed [rpm]': 'Rotational_Speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_Wear_min',
    'Machine failure': 'Machine_Failure',
    'TWF': 'Tool_Wear_Failure',
    'HDF': 'Heat_Dissipation_Failure',
    'PWF': 'Power_Failure',
    'OSF': 'Overstrain_Failure',
    'RNF': 'Random_Failure'
}, inplace=True)

# Feature Engineering
df['Delta_T'] = df['Process_Temperature_K'] - df['Air_Temperature_K']
df['Torque_per_rpm'] = df['Torque_Nm'] / (df['Rotational_Speed_rpm'] + 1)
df['Power_W'] = df['Torque_Nm'] * df['Rotational_Speed_rpm']
df['Normalized_tool_wear'] = df['Tool_Wear_min'] / df['Tool_Wear_min'].max()

# Encode categorical 'Type' column
df['Type_Encoded'] = df['Type'].astype('category').cat.codes

# Select features
features = [
    'Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_rpm',
    'Torque_Nm', 'Tool_Wear_min', 'Delta_T', 'Torque_per_rpm',
    'Power_W', 'Normalized_tool_wear', 'Type_Encoded'
]

# Define failure columns as targets
failure_cols = ['Tool_Wear_Failure', 'Heat_Dissipation_Failure', 'Power_Failure', 'Overstrain_Failure', 'Random_Failure']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features].values)

# Dictionary to store results
results = {}

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Train a model for each failure type
for failure in failure_cols:
    print(f"\n🔹 Training XGBoost for {failure}...\n")
    
    # Extract target variable
    y = df[failure].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE for imbalance handling
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Hyperparameter tuning
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)
    
    grid_search = RandomizedSearchCV(
        xgb_model, param_grid, scoring='accuracy', cv=3, n_iter=10, random_state=42, n_jobs=-1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)

    # Evaluate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy for {failure}: {accuracy:.4f}")

    # Feature importance
    feature_importances = best_model.feature_importances_
    feature_importance_dict = dict(zip(features, feature_importances))

    # Find the best feature
    best_feature = max(feature_importance_dict, key=feature_importance_dict.get)
    print(f"\n🚀 Best Feature for {failure}: {best_feature} (Importance: {feature_importance_dict[best_feature]:.4f})")

    # Save results
    results[failure] = {
        "model": best_model,
        "accuracy": accuracy,
        "feature_importances": feature_importance_dict
    }

print("\n✅ Training completed for all failure types.")
joblib.dump(model, "backend/machine_failure_model.pkl")
print("Model saved successfully!")