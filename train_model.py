import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"

TARGET_COL = "Class"
SCALE_COLS = ["Time", "Amount"]
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED_COLS = FEATURE_NAMES + [TARGET_COL]
RANDOM_STATE = 42
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]


def print_class_distribution(y, title):
    counts = y.value_counts().sort_index()
    normal_count = int(counts.get(0, 0))
    fraud_count = int(counts.get(1, 0))
    total = normal_count + fraud_count
    fraud_percentage = (fraud_count / total * 100) if total else 0.0

    print(f"\n{title}")
    print(f"Normal (Class 0): {normal_count:,}")
    print(f"Fraud  (Class 1): {fraud_count:,}")
    print(f"Fraud percentage: {fraud_percentage:.4f}%")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "creditcard.csv not found. Place it inside the data folder."
        )

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print("\nORIGINAL DATASET")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    extra_cols = [col for col in df.columns if col not in REQUIRED_COLS]
    if extra_cols:
        raise ValueError(f"Dataset contains unexpected columns: {extra_cols}")

    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    print(f"Missing values: {missing_values:,}")
    print(f"Exact duplicate rows: {duplicate_rows:,}")

    if missing_values > 0:
        missing_by_column = df.isnull().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]
        raise ValueError(
            "Dataset contains missing values:\n"
            f"{missing_by_column.to_string()}"
        )

    non_numeric_cols = df[FEATURE_NAMES].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(
            f"All input features must be numerical. Invalid columns: "
            f"{non_numeric_cols.tolist()}"
        )

    if not np.isfinite(df[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("Input features contain infinite numerical values.")

    target_values = set(df[TARGET_COL].unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(
            f"Class must contain only 0 and 1. Found: {sorted(target_values)}"
        )

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    print_class_distribution(df[TARGET_COL], "Original class distribution")

    # Remove exact duplicate rows before splitting the data
    df = df.drop_duplicates().reset_index(drop=True)

    print("\nCLEANED DATASET")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print_class_distribution(df[TARGET_COL], "Cleaned class distribution")

    return df


def prepare_data(df):
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_COL].copy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\nDATA SPLIT")
    print(f"Training set:   {len(X_train):,} rows")
    print(f"Validation set: {len(X_val):,} rows")
    print(f"Final test set: {len(X_test):,} rows")

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_fit):
    scaler = StandardScaler()
    scaler.fit(X_fit[SCALE_COLS])
    return scaler


def transform_data(X, scaler):
    X_processed = X.copy()
    X_processed[SCALE_COLS] = scaler.transform(X_processed[SCALE_COLS])
    return X_processed[FEATURE_NAMES]


def apply_smote(X, y, stage_name):
    print_class_distribution(y, f"{stage_name} - Before SMOTE")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X, y)

    if not isinstance(X_smote, pd.DataFrame):
        X_smote = pd.DataFrame(X_smote, columns=FEATURE_NAMES)
    else:
        X_smote = X_smote[FEATURE_NAMES]

    if not isinstance(y_smote, pd.Series):
        y_smote = pd.Series(y_smote, name=TARGET_COL)

    print_class_distribution(y_smote, f"{stage_name} - After SMOTE")
    return X_smote, y_smote


def build_model(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def evaluate_model(model, X_data, y_data, model_name, dataset_name):
    predictions = model.predict(X_data)

    class_labels = list(model.classes_)
    if 1 not in class_labels:
        raise ValueError(f"{model_name} does not contain fraud class 1.")

    fraud_class_index = class_labels.index(1)
    fraud_probabilities = model.predict_proba(X_data)[:, fraud_class_index]

    metrics = {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_data, predictions)),
        "Precision": float(
            precision_score(y_data, predictions, pos_label=1, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_data, predictions, pos_label=1, zero_division=0)
        ),
        "F1 Score": float(
            f1_score(y_data, predictions, pos_label=1, zero_division=0)
        ),
        "ROC-AUC": float(roc_auc_score(y_data, fraud_probabilities)),
    }

    print(f"\n{model_name.upper()} - {dataset_name.upper()} RESULTS")
    for metric_name in METRIC_NAMES:
        print(f"{metric_name}: {metrics[metric_name]:.6f}")

    matrix = confusion_matrix(y_data, predictions, labels=[0, 1])
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Fraud"],
        columns=["Predicted Normal", "Predicted Fraud"],
    )
    print("\nConfusion Matrix:")
    print(matrix_df)

    print("\nClassification Report:")
    print(
        classification_report(
            y_data,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Fraud"],
            zero_division=0,
        )
    )

    return metrics


def train_models(X_train, y_train, X_val, y_val):
    models = {
        "Random Forest": build_model("Random Forest"),
        "XGBoost": build_model("XGBoost"),
    }
    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(
            model,
            X_val,
            y_val,
            model_name,
            "Validation",
        )
        results.append(metrics)

    comparison = pd.DataFrame(results)[["Model"] + METRIC_NAMES]
    print("\nVALIDATION MODEL COMPARISON")
    print(comparison.to_string(index=False))

    return results


def select_best_model(results):
    result_by_model = {result["Model"]: result for result in results}
    rf_result = result_by_model["Random Forest"]
    xgb_result = result_by_model["XGBoost"]
    close_tolerance = 0.001

    if not np.isclose(
        rf_result["F1 Score"],
        xgb_result["F1 Score"],
        atol=close_tolerance,
    ):
        metric_used = "F1 Score"
    elif not np.isclose(
        rf_result["Recall"],
        xgb_result["Recall"],
        atol=close_tolerance,
    ):
        metric_used = "Recall"
    else:
        metric_used = "ROC-AUC"

    selected_result = max(results, key=lambda result: result[metric_used])
    selected_model_name = selected_result["Model"]

    print("\nMODEL SELECTION")
    print(f"Primary metric: F1 Score")
    print(f"Tie-break order: Recall, then ROC-AUC")
    print(f"Selection decided using: {metric_used}")
    print(f"Selected model: {selected_model_name}")

    return selected_model_name


def train_final_model(
    selected_model_name,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
):
    # Combine the original non-SMOTE training and validation samples
    X_development = pd.concat([X_train, X_val], ignore_index=True)
    y_development = pd.concat([y_train, y_val], ignore_index=True)

    # Fit a new scaler only on the combined development data
    final_scaler = fit_scaler(X_development)
    X_development_processed = transform_data(X_development, final_scaler)

    # Apply a new SMOTE instance only to the development data
    X_development_smote, y_development_smote = apply_smote(
        X_development_processed,
        y_development,
        "Final development data",
    )

    print(f"\nTraining final {selected_model_name} model...")
    final_model = build_model(selected_model_name)
    final_model.fit(X_development_smote, y_development_smote)

    # Transform the untouched test features only after final model training
    X_test_processed = transform_data(X_test, final_scaler)

    return final_model, final_scaler, X_test_processed


def save_artifacts(model, scaler, validation_results, test_metrics):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "fraud_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    feature_names_path = MODEL_DIR / "feature_names.pkl"
    model_info_path = MODEL_DIR / "model_info.json"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)
    joblib.dump(FEATURE_NAMES, feature_names_path, compress=3)

    validation_metrics = {
        result["Model"]: {
            metric_name: result[metric_name] for metric_name in METRIC_NAMES
        }
        for result in validation_results
    }
    final_test_metrics = {
        metric_name: test_metrics[metric_name] for metric_name in METRIC_NAMES
    }

    model_info = {
        "model_name": test_metrics["Model"],
        "target_column": TARGET_COL,
        "normal_class": 0,
        "fraud_class": 1,
        "scaled_columns": SCALE_COLS,
        "feature_count": len(FEATURE_NAMES),
        "selection_rule": "F1 Score, then Recall, then ROC-AUC",
        "validation_metrics": validation_metrics,
        "test_metrics": final_test_metrics,
    }

    with model_info_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    print("\nSAVED MODEL ARTIFACTS")
    print(f"Model:        {model_path}")
    print(f"Scaler:       {scaler_path}")
    print(f"Feature names:{feature_names_path}")
    print(f"Model info:   {model_info_path}")


def main():
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    print("=" * 70)

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    # Initial preprocessing uses training data only
    initial_scaler = fit_scaler(X_train)
    X_train_processed = transform_data(X_train, initial_scaler)
    X_val_processed = transform_data(X_val, initial_scaler)

    X_train_smote, y_train_smote = apply_smote(
        X_train_processed,
        y_train,
        "Training data",
    )

    validation_results = train_models(
        X_train_smote,
        y_train_smote,
        X_val_processed,
        y_val,
    )
    selected_model_name = select_best_model(validation_results)

    final_model, final_scaler, X_test_processed = train_final_model(
        selected_model_name,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
    )

    # The untouched test set is evaluated only after model selection
    final_test_metrics = evaluate_model(
        final_model,
        X_test_processed,
        y_test,
        selected_model_name,
        "Final Test",
    )

    save_artifacts(
        final_model,
        final_scaler,
        validation_results,
        final_test_metrics,
    )

    print("\nTraining and artifact saving completed successfully.")


if __name__ == "__main__":
    main()
