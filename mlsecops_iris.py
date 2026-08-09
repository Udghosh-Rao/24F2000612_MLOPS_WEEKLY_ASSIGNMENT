import os
import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

DATA_PATH = "data/iris.csv"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

TARGET = "species"

RANDOM_STATE = 42
POISONING_LEVELS = [0.00, 0.05, 0.10, 0.50]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(DATA_PATH)

print("=" * 70)
print("MLSecOps IRIS DATA POISONING EXPERIMENT")
print("=" * 70)

print("Dataset shape:", data.shape)


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

train_df, test_df = train_test_split(
    data,
    test_size=0.20,
    stratify=data[TARGET],
    random_state=RANDOM_STATE
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

X_train_clean = train_df[FEATURES].copy()
y_train_clean = train_df[TARGET].copy()

# Test data stays completely clean
X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()

print("Training samples:", len(X_train_clean))
print("Clean test samples:", len(X_test))


# ------------------------------------------------------------
# POISONING FUNCTION
# ------------------------------------------------------------

def poison_data(X, y, poisoning_rate, seed):

    X_poisoned = X.copy()
    y_poisoned = y.copy()

    rng = np.random.default_rng(seed)

    n_samples = len(X)
    n_poisoned = round(n_samples * poisoning_rate)

    if n_poisoned == 0:
        return X_poisoned, y_poisoned, 0

    indices = rng.choice(
        n_samples,
        size=n_poisoned,
        replace=False
    )

    # Replace ALL FOUR FEATURES
    feature_min = X.min(axis=0).to_numpy()
    feature_max = X.max(axis=0).to_numpy()

    random_features = rng.uniform(
        low=feature_min,
        high=feature_max,
        size=(n_poisoned, 4)
    )

    X_poisoned.iloc[indices] = random_features

    # Replace labels with random classes
    classes = sorted(y.unique())

    random_labels = rng.choice(
        classes,
        size=n_poisoned
    )

    y_poisoned.iloc[indices] = random_labels

    return X_poisoned, y_poisoned, n_poisoned


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

os.makedirs("data/mlsecops", exist_ok=True)


# ------------------------------------------------------------
# MLFLOW EXPERIMENT
# ------------------------------------------------------------

mlflow.set_experiment(
    "IRIS_MLSecOps_Data_Poisoning"
)


# ------------------------------------------------------------
# RUN 0%, 5%, 10%, 50%
# ------------------------------------------------------------

results = []

for poisoning_rate in POISONING_LEVELS:

    percentage = int(poisoning_rate * 100)

    print("\n" + "=" * 70)
    print(f"POISONING LEVEL: {percentage}%")
    print("=" * 70)

    X_train, y_train, n_poisoned = poison_data(
        X_train_clean,
        y_train_clean,
        poisoning_rate,
        seed=RANDOM_STATE + percentage
    )

    print("Training samples:", len(X_train))
    print("Poisoned samples:", n_poisoned)
    print("Clean samples:", len(X_train) - n_poisoned)

    # Save poisoned training data
    poisoned_df = X_train.copy()
    poisoned_df[TARGET] = y_train.values

    if percentage == 0:
        dataset_file = "data/mlsecops/iris_clean_train.csv"
    else:
        dataset_file = (
            f"data/mlsecops/iris_train_poisoned_{percentage}.csv"
        )

    poisoned_df.to_csv(dataset_file, index=False)

    # --------------------------------------------------------
    # MLFLOW RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"iris_poisoning_{percentage}%"
    ):

        mlflow.log_param(
            "poisoning_level",
            f"{percentage}%"
        )

        mlflow.log_param(
            "poisoning_rate",
            poisoning_rate
        )

        mlflow.log_param(
            "training_samples",
            len(X_train)
        )

        mlflow.log_param(
            "poisoned_samples",
            n_poisoned
        )

        mlflow.log_param(
            "clean_training_samples",
            len(X_train) - n_poisoned
        )

        mlflow.log_param(
            "model",
            "DecisionTreeClassifier"
        )

        mlflow.log_param(
            "max_depth",
            3
        )

        # ----------------------------------------------------
        # TRAIN SAME MODEL
        # ----------------------------------------------------

        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=1
        )

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # EVALUATE ON CLEAN TEST DATA
        # ----------------------------------------------------

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model
        model_file = (
            f"data/mlsecops/model_{percentage}.joblib"
        )

        joblib.dump(model, model_file)

        mlflow.log_artifact(model_file)
        mlflow.log_artifact(dataset_file)

        results.append({
            "poisoning": f"{percentage}%",
            "poisoned_samples": n_poisoned,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        print("\nMetrics:")
        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))


# ------------------------------------------------------------
# FINAL RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

results_df.to_csv(
    "data/mlsecops/mlsecops_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/mlsecops/mlsecops_results.csv")

print("\nMLflow experiment:")
print("IRIS_MLSecOps_Data_Poisoning")

print("\nEXPERIMENT COMPLETE")
