import json
import re
from pathlib import Path


EXPECTED_COUNTS = {
    "training": 105,
    "validation": 22,
    "test": 23,
}

VALID_SPECIES = {
    "setosa",
    "versicolor",
    "virginica",
}


def validate_file(path, split, version):
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    expected_count = EXPECTED_COUNTS[split]

    assert len(lines) == expected_count, (
        f"{path}: expected {expected_count} records, "
        f"found {len(lines)}"
    )

    for line_number, line in enumerate(lines, start=1):
        record = json.loads(line)

        assert set(record.keys()) == {
            "input_text",
            "output_text",
        }, f"{path}:{line_number}: invalid fields"

        input_text = record["input_text"]
        output_text = record["output_text"]

        assert isinstance(input_text, str)
        assert isinstance(output_text, str)
        assert input_text.strip()
        assert output_text.strip()

        if version == "v1":
            assert output_text in VALID_SPECIES, (
                f"{path}:{line_number}: "
                f"invalid v1 output: {output_text}"
            )

            pattern = (
                r"^sepal_length: \d+\.\d+, "
                r"sepal_width: \d+\.\d+, "
                r"petal_length: \d+\.\d+, "
                r"petal_width: \d+\.\d+$"
            )

            assert re.match(pattern, input_text), (
                f"{path}:{line_number}: invalid v1 input"
            )

        elif version == "v2":
            pattern = (
                r"^A flower specimen has a sepal length of "
                r"\d+\.\d+ cm, sepal width of \d+\.\d+ cm, "
                r"petal length of \d+\.\d+ cm, and petal width "
                r"of \d+\.\d+ cm\. Identify the iris species\.$"
            )

            assert re.match(pattern, input_text), (
                f"{path}:{line_number}: invalid v2 input"
            )

            expected_output = f"This is Iris {output_text.split()[-1]}."

            assert output_text.startswith("This is Iris "), (
                f"{path}:{line_number}: invalid v2 output"
            )

            species = output_text.removeprefix(
                "This is Iris "
            ).removesuffix(".")

            assert species in VALID_SPECIES, (
                f"{path}:{line_number}: "
                f"invalid v2 species: {species}"
            )


def main():
    for split in EXPECTED_COUNTS:
        for version in ["v1", "v2"]:

            if split == "training":
                directory = Path("data/training")
            elif split == "validation":
                directory = Path("data/validation")
            else:
                directory = Path("data/test")

            if version == "v1":
                filename = "iris_v1_raw.jsonl"
            else:
                filename = "iris_v2_description.jsonl"

            path = directory / filename

            validate_file(path, split, version)

            print(f"PASS: {path}")

    print("\nAll JSONL files passed validation.")


if __name__ == "__main__":
    main()
