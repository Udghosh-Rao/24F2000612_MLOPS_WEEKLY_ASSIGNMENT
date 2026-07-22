import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris_experiment")

data = pd.read_csv("data/iris.csv")
train, eval_set = train_test_split(
    data, test_size=0.2, stratify=data["species"], random_state=42
)

X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_train = train["species"]
X_eval = eval_set[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_eval = eval_set["species"]

# Task 1: vary two hyperparameters across multiple runs
param_grid = [
    {"max_depth": 2, "min_samples_split": 2},
    {"max_depth": 3, "min_samples_split": 2},
    {"max_depth": 4, "min_samples_split": 4},
    {"max_depth": 5, "min_samples_split": 4},
]

best_acc = -1
best_run_id = None

for params in param_grid:
    with mlflow.start_run() as run:
        model = DecisionTreeClassifier(random_state=1, **params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = metrics.accuracy_score(y_eval, preds)
        prec = metrics.precision_score(y_eval, preds, average="macro")
        rec = metrics.recall_score(y_eval, preds, average="macro")

        # Task 2: log params, metrics, model
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Run {run.info.run_id} | params={params} | acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_run_id = run.info.run_id

print(f"\nBest run: {best_run_id} with accuracy={best_acc:.3f}")

# Register the best model to the MLflow Model Registry (Task 5 prep)
model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri=model_uri, name="iris_classifier")
print(f"Registered model 'iris_classifier' version {result.version}")

# keep a local copy too (harmless, not tracked by dvc anymore after Task 4)
best_model = mlflow.sklearn.load_model(model_uri)
joblib.dump(best_model, "model.joblib")
