
import os
import time
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


# ---------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "train_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "model"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------
# LOAD TRAINING DATA
# ---------------------------------------

def load_training_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Training dataset not found: " + DATA_PATH
        )

    data = pd.read_csv(DATA_PATH)

    if "Class" not in data.columns:
        raise ValueError(
            "Target column 'Class' is missing."
        )

    X_train = data.drop(columns=["Class"])
    y_train = data["Class"]

    print("Training dataset loaded successfully!")
    print("Training samples:", len(X_train))
    print("Number of features:", X_train.shape[1])
    print("Number of classes:", y_train.nunique())

    return X_train, y_train


# ---------------------------------------
# DEFINE CLASSIFICATION MODELS
# ---------------------------------------

def create_models():

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))
        ]),

        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                random_state=42
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ))
        ]),

        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    random_state=42
                ),
                method="sigmoid",
                cv=3
            ))
        ])
    }

    return models


# ---------------------------------------
# TRAIN AND SAVE MODELS
# ---------------------------------------

def train_models(X_train, y_train):

    models = create_models()

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib"
    }

    training_results = []

    for model_name, model in models.items():

        print("\nTraining:", model_name)

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        file_path = os.path.join(
            MODEL_DIR,
            model_files[model_name]
        )

        joblib.dump(model, file_path)

        training_results.append({
            "Model": model_name,
            "Training Time (seconds)": round(
                training_time, 2
            ),
            "Status": "Success"
        })

        print("Training completed!")
        print("Training time:", round(
            training_time, 2
        ), "seconds")

        print("Model saved:", file_path)

    return training_results


# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------

if __name__ == "__main__":

    print("===================================")
    print("DRY BEAN CLASSIFICATION")
    print("MODEL TRAINING")
    print("===================================")

    X_train, y_train = load_training_data()

    results = train_models(X_train, y_train)

    print("\n========== TRAINING SUMMARY ==========")

    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))

    print("\nAll five models trained and saved successfully!")
