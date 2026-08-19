import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from fairlearn.metrics import MetricFrame
from scipy.stats import ks_2samp
import shap


# ============================================================
# SETUP
# ============================================================

os.makedirs("week_9/outputs", exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("WEEK 9 - EXPLAINABILITY, FAIRNESS, DRIFT AND GOVERNANCE")
print("=" * 70)


# ============================================================
# TASK 1 - LOAD DATA AND ADD LOCATION
# ============================================================

print("\nTASK 1 - LOCATION ATTRIBUTE")
print("-" * 70)

data = pd.read_csv("week_9/data/iris.csv")

data["location"] = np.random.randint(
    0,
    2,
    size=len(data)
)

print("Dataset shape:", data.shape)

print("\nLocation distribution:")
print(data["location"].value_counts().sort_index())

data.to_csv(
    "week_9/outputs/iris_with_location.csv",
    index=False
)


# ============================================================
# MODEL TRAINING
# ============================================================

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

X = data[features]
y = data["species"]

X_train, X_test, y_train, y_test, location_train, location_test = train_test_split(
    X,
    y,
    data["location"],
    test_size=0.40,
    stratify=y,
    random_state=42
)

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=1
)

model.fit(
    X_train,
    y_train
)

prediction = model.predict(X_test)


print("\nMODEL FEATURES:")
print(features)

print("\nLocation is NOT used as a model feature.")

print("\nMODEL PERFORMANCE")
print("-" * 70)

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    average="weighted",
    zero_division=0
)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        prediction,
        zero_division=0
    )
)


# ============================================================
# TASK 2 - FAIRLEARN
# ============================================================

print("\n")
print("TASK 2 - FAIRNESS WITH FAIRLEARN")
print("-" * 70)


def weighted_precision(y_true, y_pred):
    return precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


def weighted_recall(y_true, y_pred):
    return recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


metrics = {
    "accuracy": accuracy_score,
    "precision": weighted_precision,
    "recall": weighted_recall
}


metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=prediction,
    sensitive_features=location_test
)


print("\nOVERALL:")
print(metric_frame.overall)

print("\nBY LOCATION:")
print(metric_frame.by_group)

print("\nDIFFERENCE:")
print(metric_frame.difference())

print("\nRATIO:")
print(metric_frame.ratio())


metric_frame.by_group.to_csv(
    "week_9/outputs/fairness_metrics_by_location.csv"
)


# Fairness plot

fairness_data = metric_frame.by_group.reset_index()

fairness_long = fairness_data.melt(
    id_vars=["location"],
    var_name="metric",
    value_name="score"
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=fairness_long,
    x="metric",
    y="score",
    hue="location"
)

plt.ylim(0, 1.05)
plt.title("Fairness Metrics by Location")
plt.tight_layout()

plt.savefig(
    "week_9/outputs/fairness_by_location.png",
    dpi=200
)

plt.close()


# ============================================================
# TASK 3 - SHAP
# ============================================================

print("\n")
print("TASK 3 - SHAP EXPLAINABILITY")
print("-" * 70)

X_full = data[features]

explainer = shap.TreeExplainer(model)

shap_values = explainer(
    X_full
)

print("Full dataset:", X_full.shape)
print("SHAP values shape:", shap_values.values.shape)


# Setosa

plt.figure()

shap.summary_plot(
    shap_values[:, :, 0],
    X_full,
    show=False
)

plt.title("SHAP Summary - Setosa")
plt.tight_layout()

plt.savefig(
    "week_9/outputs/shap_summary_setosa.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# Versicolor

plt.figure()

shap.summary_plot(
    shap_values[:, :, 1],
    X_full,
    show=False
)

plt.title("SHAP Summary - Versicolor")
plt.tight_layout()

plt.savefig(
    "week_9/outputs/shap_summary_versicolor.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# Virginica

plt.figure()

shap.summary_plot(
    shap_values[:, :, 2],
    X_full,
    show=False
)

plt.title("SHAP Summary - Virginica")
plt.tight_layout()

plt.savefig(
    "week_9/outputs/shap_summary_virginica.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# Virginica importance

# SHAP values for Virginica
virginica_values = shap_values.values[:, :, 2]

# Mean absolute SHAP value across all 150 samples
importance_values = np.abs(
    virginica_values
).mean(axis=0)

importance = pd.DataFrame({
    "feature": features,
    "mean_absolute_shap": importance_values
})

importance = importance.sort_values(
    "mean_absolute_shap",
    ascending=False
)

print("\nVIRGINICA FEATURE IMPORTANCE:")
print(importance)

importance.to_csv(
    "week_9/outputs/virginica_shap_feature_importance.csv",
    index=False
)


# ============================================================
# TASK 4 - DATA DRIFT
# ============================================================

print("\n")
print("TASK 4 - DATA DRIFT")
print("-" * 70)

production = X_full.copy()

production["petal_length"] = (
    production["petal_length"] + 0.8
)

production["petal_width"] = (
    production["petal_width"] + 0.4
)

production["sepal_length"] = (
    production["sepal_length"] + 0.15
)


drift_results = []

for feature in features:

    statistic, p_value = ks_2samp(
        X_full[feature],
        production[feature]
    )

    drift_results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < 0.05
    })


drift_results = pd.DataFrame(
    drift_results
)

print("\nDRIFT RESULTS:")
print(drift_results.to_string(index=False))

drift_results.to_csv(
    "week_9/outputs/drift_results.csv",
    index=False
)


# Drift plots

for feature in features:

    plt.figure(figsize=(8, 5))

    sns.kdeplot(
        X_full[feature],
        label="Training"
    )

    sns.kdeplot(
        production[feature],
        label="Production"
    )

    plt.title(
        "Training vs Production - " + feature
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "week_9/outputs/drift_" + feature + ".png",
        dpi=200
    )

    plt.close()


# ============================================================
# TASK 5 - MODEL CARD
# ============================================================

print("\n")
print("TASK 5 - MODEL CARD")
print("-" * 70)


model_card = f"""# Model Card - IRIS Classifier

## Model Overview

Decision Tree classifier for the Iris dataset.

## Intended Use

Educational demonstration of:

- Machine learning classification
- Explainability
- Fairness
- Data drift
- ML governance

The model is not intended for high-stakes decisions.

## Training Data

The model uses the Iris dataset with four features:

- sepal_length
- sepal_width
- petal_length
- petal_width

A random location attribute with values 0 and 1 was added for
fairness analysis.

Location was NOT used as a model training feature.

## Overall Performance

Accuracy: {accuracy:.4f}

Precision: {precision:.4f}

Recall: {recall:.4f}

## Fairness

Fairlearn MetricFrame was used to evaluate accuracy, precision
and recall separately for location groups 0 and 1.

Location was randomly assigned, so large systematic performance
differences are not expected.

## Explainability

SHAP was used with the full Iris dataset.

Summary plots were generated for:

- Setosa
- Versicolor
- Virginica

Positive SHAP values push predictions toward the selected class.

Negative SHAP values push predictions away from the selected class.

Red points represent relatively high feature values.

Blue points represent relatively low feature values.

## Drift

Production data was simulated by shifting feature distributions.

The Kolmogorov-Smirnov test was used to detect distribution drift.

A p-value below 0.05 indicates statistically significant drift.

## Limitations

- Iris is a small educational dataset.
- Location is randomly generated.
- Fairness results demonstrate the auditing method.
- Data drift does not automatically mean concept drift.
- The model is not appropriate for high-stakes decisions.

## Governance

Training data, model versions, performance metrics, fairness
results, explainability results and drift results should be
tracked and retained for responsible ML deployment.
"""


with open(
    "week_9/MODEL_CARD.md",
    "w"
) as f:

    f.write(model_card)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("WEEK 9 ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

for root, dirs, files in os.walk("week_9/outputs"):

    for file in sorted(files):

        print(
            os.path.join(root, file)
        )

print("\nModel card:")
print("week_9/MODEL_CARD.md")
