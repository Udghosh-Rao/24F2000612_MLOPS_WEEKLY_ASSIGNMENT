# Model Card - IRIS Classifier

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

Accuracy: 0.9500

Precision: 0.9507

Recall: 0.9500

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
