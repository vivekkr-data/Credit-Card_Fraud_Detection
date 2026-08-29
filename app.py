import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

SCALE_COLS = ["Time", "Amount"]
EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_FEATURE_VALUE = np.finfo(np.float32).max


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Please run train_model.py first."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Scaler not found. Please run train_model.py first."
        )
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Feature information not found. Please train the model first."
        )
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            "Model information not found. Please run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    with MODEL_INFO_PATH.open("r", encoding="utf-8") as file:
        model_info = json.load(file)

    if feature_names != EXPECTED_FEATURES:
        raise ValueError(
            "Saved feature names or their order are invalid. Please retrain the model."
        )
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Saved model is invalid. Please retrain the model.")
    if not callable(getattr(scaler, "transform", None)):
        raise ValueError("Saved scaler is invalid. Please retrain the model.")
    if getattr(model, "n_features_in_", None) != len(feature_names):
        raise ValueError("Saved model has an invalid feature count. Please retrain it.")
    if getattr(scaler, "n_features_in_", None) != len(SCALE_COLS):
        raise ValueError("Saved scaler has an invalid feature count. Please retrain it.")
    model_classes = getattr(model, "classes_", None)
    if model_classes is None or set(model_classes) != {0, 1}:
        raise ValueError("Saved model does not contain the expected classes 0 and 1.")
    if not isinstance(model_info, dict):
        raise ValueError("Saved model information is invalid. Please retrain the model.")

    return model, scaler, feature_names, model_info


def preprocess_data(data, scaler, feature_names):
    processed = data[feature_names].copy()
    processed[SCALE_COLS] = scaler.transform(processed[SCALE_COLS])
    return processed[feature_names]


def predict_transactions(model, processed_data):
    predictions = np.asarray(model.predict(processed_data), dtype=int)
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("The trained model returned an unexpected class label.")

    fraud_scores = None
    if callable(getattr(model, "predict_proba", None)):
        class_labels = list(model.classes_)
        if 1 not in class_labels:
            raise ValueError("The trained model does not contain fraud class 1.")
        fraud_class_index = class_labels.index(1)
        fraud_scores = np.asarray(
            model.predict_proba(processed_data)[:, fraud_class_index],
            dtype=float,
        )
        if (
            not np.isfinite(fraud_scores).all()
            or (fraud_scores < 0).any()
            or (fraud_scores > 1).any()
        ):
            raise ValueError("The trained model returned an invalid fraud score.")

    return predictions, fraud_scores


def show_single_prediction(model, scaler, feature_names):
    st.subheader("Single Transaction Prediction")
    st.write(
        "Enter one transaction record. Time is elapsed transaction time, "
        "Amount is the transaction amount, and V1–V28 are anonymized PCA features."
    )
    st.info(
        "The displayed zeros are empty-form defaults, not a sample transaction. "
        "Enter values from a real transaction record to enable prediction."
    )

    top_col_1, top_col_2 = st.columns(2)
    with top_col_1:
        time_value = st.number_input(
            "Time",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.6f",
            help="Elapsed transaction time from the dataset reference.",
        )
    with top_col_2:
        amount_value = st.number_input(
            "Amount",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.6f",
            help="Transaction amount.",
        )

    input_values = {"Time": time_value, "Amount": amount_value}

    with st.expander("Enter V1–V28 PCA Features", expanded=True):
        st.caption(
            "These features are anonymized. Their original business meanings are not available."
        )
        feature_columns = st.columns(4)
        pca_features = [name for name in feature_names if name not in SCALE_COLS]

        for index, feature_name in enumerate(pca_features):
            with feature_columns[index % 4]:
                input_values[feature_name] = st.number_input(
                    feature_name,
                    value=0.0,
                    step=0.1,
                    format="%.6f",
                    key=f"single_{feature_name}",
                )

    has_transaction_values = any(value != 0.0 for value in input_values.values())
    if st.button(
        "Predict Transaction",
        type="primary",
        width="stretch",
        disabled=not has_transaction_values,
        help=(
            "Enter at least one non-zero transaction value first."
            if not has_transaction_values
            else None
        ),
    ):
        try:
            input_df = validate_input_data(
                pd.DataFrame([input_values]),
                feature_names,
            )
            input_processed = preprocess_data(input_df, scaler, feature_names)
            predictions, fraud_scores = predict_transactions(model, input_processed)

            prediction = int(predictions[0])
            if prediction == 0:
                st.success("Normal Transaction")
                st.write("This transaction was classified as normal by the model.")
            else:
                st.error("Potential Fraud Detected")
                st.write(
                    "This transaction was classified as potentially fraudulent by the model."
                )

            if fraud_scores is not None:
                st.metric("Fraud Score", f"{float(fraud_scores[0]) * 100:.2f}%")
                st.caption(
                    "This score is produced by the machine learning model and should not "
                    "be interpreted as a calibrated real-world banking risk probability."
                )
        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")


def validate_input_data(input_data, feature_names):
    if input_data.empty:
        raise ValueError("The uploaded CSV file does not contain any transaction rows.")

    missing_cols = [col for col in feature_names if col not in input_data.columns]
    if missing_cols:
        raise ValueError(
            "Uploaded CSV is missing the following required columns: "
            + ", ".join(missing_cols)
        )

    numeric_data = input_data[feature_names].apply(pd.to_numeric, errors="coerce")
    invalid_cols = numeric_data.columns[numeric_data.isnull().any()].tolist()
    if invalid_cols:
        raise ValueError(
            "The following required columns contain missing or non-numerical values: "
            + ", ".join(invalid_cols)
        )

    numeric_values = numeric_data.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Required feature columns contain infinite numerical values.")
    if (np.abs(numeric_values) > MAX_FEATURE_VALUE).any():
        raise ValueError(
            "Required feature columns contain values outside the supported "
            "numerical range."
        )

    return numeric_data[feature_names]


def show_batch_prediction(model, scaler, feature_names):
    st.subheader("Batch CSV Prediction")
    st.write(
        "Upload a CSV containing Time, V1–V28, and Amount. A Class column or other "
        "extra columns are allowed and will be preserved in the downloaded results."
    )

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a transaction CSV file to begin batch prediction.")
        return

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError):
        st.error("Please upload a valid CSV file.")
        return

    try:
        model_input = validate_input_data(uploaded_data, feature_names)
        processed_data = preprocess_data(model_input, scaler, feature_names)
        predictions, fraud_scores = predict_transactions(model, processed_data)
    except Exception as error:
        st.error(str(error))
        return

    result_data = uploaded_data.copy()
    result_data["Prediction"] = np.where(predictions == 1, "Fraud", "Normal")
    if fraud_scores is not None:
        result_data["Fraud_Score"] = fraud_scores

    total_transactions = len(result_data)
    fraud_transactions = int(np.sum(predictions == 1))
    normal_transactions = total_transactions - fraud_transactions
    predicted_fraud_rate = fraud_transactions / total_transactions * 100

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Total Transactions", f"{total_transactions:,}")
    metric_col_2.metric("Normal Transactions", f"{normal_transactions:,}")
    metric_col_3.metric("Fraudulent Transactions", f"{fraud_transactions:,}")
    metric_col_4.metric("Predicted Fraud Rate", f"{predicted_fraud_rate:.2f}%")

    st.dataframe(result_data, width="stretch")

    prediction_csv = result_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction Results",
        data=prediction_csv,
        file_name="fraud_prediction_results.csv",
        mime="text/csv",
        width="stretch",
    )


def show_about_model(model_info):
    st.subheader("About Model")

    selected_model = model_info.get("model_name", "Not available")
    st.write(f"**Selected Model:** {selected_model}")
    st.caption(
        "The metrics below are fixed evaluation results from the complete held-out "
        "test set. They are not calculated from the transaction entered in the form."
    )

    test_metrics = model_info.get("test_metrics", {})
    metric_columns = st.columns(5)
    for column, metric_name in zip(metric_columns, [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
    ]):
        value = test_metrics.get(metric_name)
        try:
            displayed_value = f"{float(value):.4f}" if value is not None else "N/A"
        except (TypeError, ValueError):
            displayed_value = "N/A"
        column.metric(f"Test {metric_name}", displayed_value)

    st.markdown(
        """
### Methodology

- The dataset is highly imbalanced, so Accuracy is not used alone.
- Exact duplicate rows are removed before splitting.
- During model comparison, the scaler is fitted only on training data.
- SMOTE is applied only to training data during comparison and combined development data during final retraining.
- Random Forest and XGBoost are compared on a validation set.
- The final test set remains untouched until the selected model's final evaluation.

### Limitations

- V1–V28 are anonymized, which limits interpretability.
- Fraud patterns can change over time.
- This application is an educational machine learning project and not a production banking decision system.
"""
    )


def main():
    st.title("Credit Card Fraud Detection System")
    st.write(
        "Machine Learning based fraud detection using SMOTE, Random Forest and XGBoost."
    )
    st.caption(
        "This application uses a trained machine learning model to classify credit card "
        "transaction records as Normal or potentially Fraudulent. V1–V28 are anonymized "
        "PCA-derived features from the dataset."
    )

    try:
        model, scaler, feature_names, model_info = load_artifacts()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Train the model locally with: python train_model.py")
        st.stop()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Saved model artifacts could not be loaded: {error}")
        st.info("Please run train_model.py again to recreate the model artifacts.")
        st.stop()
    except Exception:
        st.error(
            "Saved model artifacts are incompatible or damaged. "
            "Please run train_model.py again."
        )
        st.stop()

    with st.sidebar:
        st.header("Project Information")
        st.write("**Project:** Credit Card Fraud Detection")
        st.write("**Models Compared:** Random Forest and XGBoost")
        st.write("**Imbalance Handling:** SMOTE")
        st.write(f"**Selected Model:** {model_info.get('model_name', 'Not available')}")
        st.write("**Deployment:** Streamlit")

    single_tab, batch_tab, about_tab = st.tabs(
        ["Single Transaction", "Batch CSV Prediction", "About Model"]
    )

    with single_tab:
        show_single_prediction(model, scaler, feature_names)

    with batch_tab:
        show_batch_prediction(model, scaler, feature_names)

    with about_tab:
        show_about_model(model_info)


if __name__ == "__main__":
    main()
