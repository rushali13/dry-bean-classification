
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, file_name in MODEL_FILES.items():

        model_path = os.path.join(MODEL_DIR, file_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {file_name}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=model.classes_
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }, y_pred


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🌱 Dry Bean Classification Dashboard")

st.markdown(
    """
    ### Machine Learning Model Evaluation

    This application compares five machine learning
    classification algorithms on the UCI Dry Bean dataset.

    Upload your test dataset, select a classifier and
    explore its evaluation performance.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

st.sidebar.info(
    """
    **Available Models**

    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors
    4. Gaussian Naive Bayes
    5. Support Vector Machine
    """
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

try:
    models = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.subheader("📁 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_data = pd.read_csv(uploaded_file)

        st.write("### Dataset Preview")
        st.dataframe(test_data.head(), use_container_width=True)

        st.write(
            f"**Total records:** {test_data.shape[0]}"
        )

        # ------------------------------------------
        # VALIDATE TARGET COLUMN
        # ------------------------------------------

        if "Class" not in test_data.columns:
            st.error(
                "The uploaded CSV must contain a 'Class' target column."
            )
            st.stop()

        # ------------------------------------------
        # SEPARATE FEATURES AND TARGET
        # ------------------------------------------

        y_test = test_data["Class"]
        X_test = test_data.drop(columns=["Class"])

        # ------------------------------------------
        # VALIDATE FEATURE COLUMNS
        # ------------------------------------------

        expected_features = list(
            models["Logistic Regression"].feature_names_in_
        )

        missing_features = [
            col for col in expected_features
            if col not in X_test.columns
        ]

        extra_features = [
            col for col in X_test.columns
            if col not in expected_features
        ]

        if missing_features:
            st.error(
                f"Missing required features: {missing_features}"
            )
            st.stop()

        if extra_features:
            st.warning(
                f"Extra columns will be ignored: {extra_features}"
            )

        X_test = X_test[expected_features]

        # ------------------------------------------
        # VALIDATE DATA
        # ------------------------------------------

        if X_test.isnull().any().any() or y_test.isnull().any():
            st.error(
                "The uploaded dataset contains missing values. "
                "Please upload a cleaned test CSV."
            )
            st.stop()

        if not all(
            pd.api.types.is_numeric_dtype(X_test[col])
            for col in X_test.columns
        ):
            st.error("All 16 input features must be numerical.")
            st.stop()

        if len(X_test) == 0:
            st.error("The uploaded dataset is empty.")
            st.stop()

        # ------------------------------------------
        # CHECK TARGET CLASSES
        # ------------------------------------------

        expected_classes = set(
            models[selected_model].classes_
        )

        uploaded_classes = set(y_test.unique())

        unknown_classes = uploaded_classes - expected_classes

        if unknown_classes:
            st.error(
                f"Unknown target classes found: {unknown_classes}"
            )
            st.stop()

        st.success(
            "Dataset validation completed successfully!"
        )

        st.divider()

        # ------------------------------------------
        # EVALUATE ALL FIVE MODELS
        # ------------------------------------------

        st.subheader("📊 Model Performance Comparison")

        all_results = {}

        for model_name, model in models.items():

            metrics, predictions = evaluate_model(
                model,
                X_test,
                y_test
            )

            all_results[model_name] = metrics

        comparison_df = pd.DataFrame(
            all_results
        ).T

        st.dataframe(
            comparison_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # ------------------------------------------
        # SELECTED MODEL METRICS
        # ------------------------------------------

        st.divider()

        st.subheader(
            f"📈 Evaluation Metrics – {selected_model}"
        )

        selected_metrics = all_results[selected_model]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{selected_metrics['Accuracy']:.4f}"
        )

        col2.metric(
            "AUC Score",
            f"{selected_metrics['AUC']:.4f}"
        )

        col3.metric(
            "Precision",
            f"{selected_metrics['Precision']:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{selected_metrics['Recall']:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{selected_metrics['F1 Score']:.4f}"
        )

        col6.metric(
            "MCC Score",
            f"{selected_metrics['MCC']:.4f}"
        )

        # ------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------

        st.divider()

        st.subheader("🎯 Confusion Matrix")

        selected_model_object = models[selected_model]

        y_pred = selected_model_object.predict(X_test)

        class_names = selected_model_object.classes_

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=class_names
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"Confusion Matrix – {selected_model}")

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ------------------------------------------
        # CLASSIFICATION REPORT
        # ------------------------------------------

        st.divider()

        st.subheader("📋 Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=class_names,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # ------------------------------------------
        # DOWNLOAD RESULTS
        # ------------------------------------------

        st.divider()

        st.subheader("⬇️ Download Evaluation Results")

        csv_data = comparison_df.to_csv().encode("utf-8")

        st.download_button(
            label="Download Model Comparison CSV",
            data=csv_data,
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Unable to process the uploaded dataset: {e}")

else:

    st.info(
        "Please upload test_data.csv to view model evaluation results."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Assignment 2 | M.Tech AIML | "
    "BITS Pilani | Dry Bean Classification"
)
