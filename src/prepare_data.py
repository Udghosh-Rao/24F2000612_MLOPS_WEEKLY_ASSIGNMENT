import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "training"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"


def make_record_v1(row):
    return {
        "input_text": (
            f"sepal_length: {row['sepal_length']:.1f}, "
            f"sepal_width: {row['sepal_width']:.1f}, "
            f"petal_length: {row['petal_length']:.1f}, "
            f"petal_width: {row['petal_width']:.1f}"
        ),
        "output_text": row["species"],
    }


def make_record_v2(row):
    return {
        "input_text": (
            f"A flower specimen has a sepal length of "
            f"{row['sepal_length']:.1f} cm, sepal width of "
            f"{row['sepal_width']:.1f} cm, petal length of "
            f"{row['petal_length']:.1f} cm, and petal width of "
            f"{row['petal_width']:.1f} cm. Identify the iris species."
        ),
        "output_text": f"This is Iris {row['species']}.",
    }


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def prepare_dataset():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    df["species"] = [
        iris.target_names[index]
        for index in iris.target
    ]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["species"],
        random_state=RANDOM_STATE,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["species"],
        random_state=RANDOM_STATE,
    )

    datasets = {
        "training": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for split_name, split_df in datasets.items():

        v1_records = [
            make_record_v1(row)
            for _, row in split_df.iterrows()
        ]

        v2_records = [
            make_record_v2(row)
            for _, row in split_df.iterrows()
        ]

        if split_name == "training":
            v1_path = TRAIN_DIR / "iris_v1_raw.jsonl"
            v2_path = TRAIN_DIR / "iris_v2_description.jsonl"

        elif split_name == "validation":
            v1_path = VALIDATION_DIR / "iris_v1_raw.jsonl"
            v2_path = VALIDATION_DIR / "iris_v2_description.jsonl"

        else:
            v1_path = TEST_DIR / "iris_v1_raw.jsonl"
            v2_path = TEST_DIR / "iris_v2_description.jsonl"

        write_jsonl(v1_records, v1_path)
        write_jsonl(v2_records, v2_path)

        print(f"{split_name}: {len(split_df)} samples")


if __name__ == "__main__":
    prepare_dataset()
    print("Dataset preparation complete.")
