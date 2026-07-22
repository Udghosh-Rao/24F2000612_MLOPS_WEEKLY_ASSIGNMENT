import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

mlflow.set_tracking_uri("http://localhost:5000")

data = pd.read_csv("data/iris.csv")
_, eval_set = train_test_split(
    data, test_size=0.2, stratify=data["species"], random_state=42
)
X_eval = eval_set[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_eval = eval_set["species"]

model = mlflow.sklearn.load_model("models:/iris_classifier/latest")

preds = model.predict(X_eval)
acc = metrics.accuracy_score(y_eval, preds)
prec = metrics.precision_score(y_eval, preds, average="macro")
rec = metrics.recall_score(y_eval, preds, average="macro")

print(f"Loaded model from MLflow Registry: iris_classifier/latest")
print(f"Eval accuracy: {acc:.3f} | precision: {prec:.3f} | recall: {rec:.3f}")
