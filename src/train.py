# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import numpy as np

# 1. Load the Processed Data
print("Loading data...")
df = pd.read_csv('data/processed_diabetic_data.csv')

# For this baseline, we will drop rows with any remaining NaNs for simplicity.
# A more advanced approach would involve more sophisticated imputation.
df.dropna(inplace=True)

# 2. Define Features (X) and Target (y)
# Let's select a mix of numeric and categorical features from our processed data
numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']
categorical_features = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1']

# Ensure all feature columns are present (Defensive Programming)
for col in numeric_features + categorical_features:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the dataframe.")

X = df[numeric_features + categorical_features]
y = df['target']

# 3. Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Define the Preprocessing Pipeline
# This object applies different transformations to different columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# 5. Set up MLflow Experiment
mlflow.set_experiment("Diabetes Readmission Prediction")

# 6. Define a function to evaluate the model
def eval_metrics(actual, pred):
    recall = recall_score(actual, pred)
    precision = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    return recall, precision, f1, accuracy

# 7. Train and Log the Baseline Model (Logistic Regression)
with mlflow.start_run(run_name="LogisticRegression_Baseline"):
    print("Training Logistic Regression model...")

    # Create the full pipeline with the preprocessor and the classifier
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Train the model pipeline
    lr_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr_pipeline.predict(X_test)

    # Evaluate metrics using our helper function
    (recall, precision, f1, accuracy) = eval_metrics(y_test, y_pred)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(lr_pipeline, "model")

    # Print results to the console
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

print("\nTraining script finished.")


# 8. Train and Log the Second Model (Random Forest)
with mlflow.start_run(run_name="RandomForest_Baseline"):
    print("\nTraining Random Forest model...")

    # Create the full pipeline with the preprocessor and the classifier
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])

    # Train the model
    rf_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_pipeline.predict(X_test)

    # Evaluate metrics
    (recall_rf, precision_rf, f1_rf, accuracy_rf) = eval_metrics(y_test, y_pred_rf)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_metric("recall", recall_rf)
    mlflow.log_metric("precision", precision_rf)
    mlflow.log_metric("f1_score", f1_rf)
    mlflow.log_metric("accuracy", accuracy_rf)

    mlflow.sklearn.log_model(rf_pipeline, "model")

    print(f"  Recall: {recall_rf:.4f}")
    print(f"  Precision: {precision_rf:.4f}")
    print(f"  F1-score: {f1_rf:.4f}")
    print(f"  Accuracy: {accuracy_rf:.4f}")


print("\nTraining script finished.")


# 9. Train and Log the Third Model (XGBoost)
with mlflow.start_run(run_name="XGBoost_Baseline"):
    print("\nTraining XGBoost model...")

    # XGBoost has a specific parameter for handling class imbalance
    # It's the ratio of the number of negative class to the positive class
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"  XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    # Create the full pipeline with the preprocessor and the classifier
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ))
    ])

    # Train the model
    xgb_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred_xgb = xgb_pipeline.predict(X_test)

    # Evaluate metrics
    (recall_xgb, precision_xgb, f1_xgb, accuracy_xgb) = eval_metrics(y_test, y_pred_xgb)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("recall", recall_xgb)
    mlflow.log_metric("precision", precision_xgb)
    mlflow.log_metric("f1_score", f1_xgb)
    mlflow.log_metric("accuracy", accuracy_xgb)

    mlflow.sklearn.log_model(xgb_pipeline, "model")

    print(f"  Recall: {recall_xgb:.4f}")
    print(f"  Precision: {precision_xgb:.4f}")
    print(f"  F1-score: {f1_xgb:.4f}")
    print(f"  Accuracy: {accuracy_xgb:.4f}")


print("\nTraining script finished.")