import joblib
import pandas as pd
import pytest
from sklearn import metrics
from sklearn.model_selection import train_test_split

DATA_PATH = "data/iris.csv"
MODEL_PATH = "model.joblib"

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL_COLUMN = "species"

# Minimum acceptable scores. IRIS is an easy dataset with a simple decision
# tree in train.py, so a healthy model should comfortably clear these.
MIN_ACCURACY = 0.85
MIN_MACRO_F1 = 0.80


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def eval_split():
    """
    Rebuild the SAME eval split train.py used (same random_state=42,
    test_size=0.2) so we're scoring on genuinely held-out rows, not data
    the model was trained on.
    """
    data = pd.read_csv(DATA_PATH)
    _, eval_set = train_test_split(
        data, test_size=0.2, stratify=data[LABEL_COLUMN], random_state=42
    )
    X_eval = eval_set[FEATURE_COLUMNS]
    y_eval = eval_set[LABEL_COLUMN]
    return X_eval, y_eval


def test_model_file_loads(model):
    """The model artifact must be a valid, loadable sklearn model."""
    assert model is not None
    assert hasattr(model, "predict"), "Loaded object has no predict() method"


def test_model_predicts_known_classes(model, eval_split):
    """Predictions should only ever be one of the three IRIS species."""
    X_eval, y_eval = eval_split
    preds = model.predict(X_eval)
    valid_classes = set(y_eval.unique())
    assert set(preds).issubset(valid_classes), (
        f"Model predicted unexpected class(es): {set(preds) - valid_classes}"
    )


def test_model_meets_minimum_accuracy(model, eval_split):
    """Fails CI if accuracy on the eval set drops below the quality bar."""
    X_eval, y_eval = eval_split
    preds = model.predict(X_eval)
    acc = metrics.accuracy_score(y_eval, preds)
    assert acc >= MIN_ACCURACY, (
        f"Eval accuracy {acc:.3f} is below the minimum threshold {MIN_ACCURACY}"
    )


def test_model_meets_minimum_macro_f1(model, eval_split):
    """
    Macro F1 catches the case where accuracy looks fine overall but the
    model is quietly failing on one specific species.
    """
    X_eval, y_eval = eval_split
    preds = model.predict(X_eval)
    f1 = metrics.f1_score(y_eval, preds, average="macro")
    assert f1 >= MIN_MACRO_F1, (
        f"Macro F1 {f1:.3f} is below the minimum threshold {MIN_MACRO_F1}"
    )


def test_no_missing_predictions(model, eval_split):
    """Every eval row must get exactly one prediction back."""
    X_eval, y_eval = eval_split
    preds = model.predict(X_eval)
    assert len(preds) == len(y_eval), "Prediction count does not match eval set size"