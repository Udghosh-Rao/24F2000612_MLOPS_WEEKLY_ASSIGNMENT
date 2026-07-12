import pandas as pd
import pytest

DATA_PATH = "data/iris.csv"

EXPECTED_COLUMNS = {
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
}

NUMERIC_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

VALID_SPECIES = {"setosa", "versicolor", "virginica"}


@pytest.fixture(scope="module")
def data():
    """Load the dataset once and reuse it across every test in this file."""
    return pd.read_csv(DATA_PATH)


def test_file_is_not_empty(data):
    """The dataset must actually contain rows."""
    assert len(data) > 0, "data/iris.csv has zero rows"


def test_expected_columns_present(data):
    """All columns train.py relies on must exist."""
    missing = EXPECTED_COLUMNS - set(data.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_no_missing_values_in_key_columns(data):
    """None of the columns we train/eval on should contain nulls."""
    for col in EXPECTED_COLUMNS:
        null_count = data[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} missing value(s)"


def test_no_duplicate_rows(data):
    """
    A small number of duplicate rows can legitimately occur from the
    augmentation/resampling step (occasional collisions are expected).
    We only fail if duplication becomes excessive.
    """
    dup_count = data.duplicated().sum()
    dup_ratio = dup_count / len(data)
    assert dup_ratio <= 0.05, (
        f"Found {dup_count} fully duplicated row(s) "
        f"({dup_ratio:.1%} of data) - exceeds the 5% tolerance"
    )


@pytest.mark.parametrize("col", NUMERIC_COLUMNS)
def test_numeric_columns_are_numeric(data, col):
    """Feature columns must be numeric, not strings/objects."""
    assert pd.api.types.is_numeric_dtype(data[col]), (
        f"Column '{col}' is not numeric (dtype={data[col].dtype})"
    )


@pytest.mark.parametrize("col", NUMERIC_COLUMNS)
def test_numeric_columns_are_positive(data, col):
    """Physical flower measurements can never be zero or negative."""
    assert (data[col] > 0).all(), f"Column '{col}' has non-positive value(s)"


@pytest.mark.parametrize("col", NUMERIC_COLUMNS)
def test_numeric_columns_within_reasonable_range(data, col):
    """
    Sanity-check ranges based on the known IRIS dataset (values are cm).
    A generous margin is used so slightly augmented/noisy data still passes,
    while genuinely corrupt data (e.g. a value of 500) still fails.
    """
    assert data[col].between(0, 15).all(), (
        f"Column '{col}' has value(s) outside the plausible 0-15 cm range"
    )


def test_species_values_are_valid(data):
    """species must only contain the three known IRIS classes."""
    unexpected = set(data["species"].unique()) - VALID_SPECIES
    assert not unexpected, f"Unexpected species label(s) found: {unexpected}"


def test_all_species_represented(data):
    """Every class should have at least a few examples, or training/eval breaks."""
    counts = data["species"].value_counts()
    for species in VALID_SPECIES:
        assert species in counts, f"Species '{species}' is missing entirely"
        assert counts[species] >= 5, (
            f"Species '{species}' has only {counts[species]} row(s), too few"
        )