import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "project-83abca74-808e-4448-be0"
LOCATION = "us-central1"

V1_ENDPOINT = "4753378432631439360"
V2_ENDPOINT = "6035074739083411456"

V1_TEST_FILE = "data/test/iris_v1_raw.jsonl"
V2_TEST_FILE = "data/test/iris_v2_description.jsonl"

RESULTS_DIR = "evaluation/results"

SPECIES = ["setosa", "versicolor", "virginica"]

MAX_OUTPUT_TOKENS = 100
TEMPERATURE = 0

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# AUTHENTICATION
# ============================================================

def get_access_token():
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


# ============================================================
# GEMINI INFERENCE
# ============================================================

def generate_content(endpoint_id, prompt, token):
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"endpoints/{endpoint_id}:generateContent"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")

        data = json.loads(body)

        text = ""

        try:
            parts = data["candidates"][0]["content"]["parts"]

            text = "".join(
                part.get("text", "")
                for part in parts
            )

        except (KeyError, IndexError, TypeError):
            text = ""

        finish_reason = None

        try:
            finish_reason = data["candidates"][0].get(
                "finishReason"
            )
        except (KeyError, IndexError, TypeError):
            pass

        return {
            "text": text.strip(),
            "finish_reason": finish_reason,
            "raw_response": data,
            "error": None,
        }

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")

        return {
            "text": "",
            "finish_reason": None,
            "raw_response": None,
            "error": f"HTTP {e.code}: {body}",
        }

    except Exception as e:
        return {
            "text": "",
            "finish_reason": None,
            "raw_response": None,
            "error": str(e),
        }


# ============================================================
# SPECIES EXTRACTION
# ============================================================

def extract_species(text):
    """
    Extract a species name from the model response.

    This is used ONLY for classification metrics.

    Format compliance is calculated separately.
    """

    if not text:
        return None

    normalized = text.lower()

    # Most specific first.
    for species in SPECIES:
        if re.search(
            rf"\biris\s+{species}\b",
            normalized,
        ):
            return species

    # Also accept plain species names.
    for species in SPECIES:
        if re.search(
            rf"\b{species}\b",
            normalized,
        ):
            return species

    return None


# ============================================================
# FORMAT COMPLIANCE
# ============================================================

def check_format_v1(text):
    """
    V1 expected output:

        setosa
        versicolor
        virginica

    Exact match only.
    """

    if not text:
        return False

    return text in SPECIES


def check_format_v2(text):
    """
    V2 expected output:

        This is Iris setosa.
        This is Iris versicolor.
        This is Iris virginica.

    Exact format only.
    """

    if not text:
        return False

    pattern = (
        r"^This is Iris "
        r"(setosa|versicolor|virginica)"
        r"\.$"
    )

    return bool(re.fullmatch(pattern, text))


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(records):
    confusion = {
        actual: {
            predicted: 0
            for predicted in SPECIES
        }
        for actual in SPECIES
    }

    per_class = {}

    for species in SPECIES:

        tp = 0
        fp = 0
        fn = 0

        for record in records:

            actual = record["expected"]
            predicted = record["predicted"]

            if actual == species and predicted == species:
                tp += 1

            elif actual != species and predicted == species:
                fp += 1

            elif actual == species and predicted != species:
                fn += 1

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        per_class[species] = {
            "precision": precision,
            "recall": recall,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    for record in records:

        actual = record["expected"]
        predicted = record["predicted"]

        if actual in SPECIES and predicted in SPECIES:
            confusion[actual][predicted] += 1

    accuracy = (
        sum(
            1
            for r in records
            if r["predicted"] == r["expected"]
        )
        / len(records)
        if records
        else 0.0
    )

    format_compliance = (
        sum(
            1
            for r in records
            if r["format_compliant"]
        )
        / len(records)
        if records
        else 0.0
    )

    macro_precision = (
        sum(
            per_class[s]["precision"]
            for s in SPECIES
        )
        / len(SPECIES)
    )

    macro_recall = (
        sum(
            per_class[s]["recall"]
            for s in SPECIES
        )
        / len(SPECIES)
    )

    return {
        "accuracy": accuracy,
        "format_compliance": format_compliance,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    name,
    endpoint,
    test_file,
    format_checker,
    token,
):

    print()
    print("=" * 70)
    print(f"EVALUATING {name}")
    print("=" * 70)

    records = load_jsonl(test_file)

    print(f"Endpoint: {endpoint}")
    print(f"Test file: {test_file}")
    print(f"Test examples: {len(records)}")
    print()

    results = []

    for i, example in enumerate(records, 1):

        prompt = example["input_text"]
        expected_output = example["output_text"]

        expected_species = extract_species(
            expected_output
        )

        response = generate_content(
            endpoint,
            prompt,
            token,
        )

        output = response["text"]

        predicted_species = extract_species(
            output
        )

        format_compliant = format_checker(
            output
        )

        correct = (
            predicted_species == expected_species
        )

        result = {
            "index": i,
            "input": prompt,
            "expected_output": expected_output,
            "expected_species": expected_species,
            "output": output,
            "predicted": predicted_species,
            "correct": correct,
            "format_compliant": format_compliant,
            "finish_reason": response["finish_reason"],
            "error": response["error"],
        }

        results.append(result)

        print(
            f"[{i:02d}/{len(records)}] "
            f"Expected={expected_species}"
        )

        print(
            f"    Output={output!r}"
        )

        print(
            f"    Predicted={predicted_species} "
            f"Correct={correct} "
            f"Format={format_compliant}"
        )

        if response["error"]:
            print(
                f"    ERROR={response['error']}"
            )

        # Small delay to avoid unnecessary request bursts.
        time.sleep(0.2)

    metrics_input = [
        {
            "expected": r["expected_species"],
            "predicted": r["predicted"],
            "format_compliant": r["format_compliant"],
        }
        for r in results
    ]

    metrics = calculate_metrics(
        metrics_input
    )

    output_data = {
        "model": name,
        "endpoint": endpoint,
        "test_file": test_file,
        "test_examples": len(results),
        "metrics": metrics,
        "predictions": results,
    }

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_evaluation.json",
    )

    with open(output_path, "w") as f:
        json.dump(
            output_data,
            f,
            indent=2,
        )

    print()
    print("-" * 70)
    print(f"{name} RESULTS")
    print("-" * 70)

    print(
        f"Accuracy:              "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Format compliance:     "
        f"{metrics['format_compliance'] * 100:.2f}%"
    )

    print(
        f"Macro precision:       "
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Macro recall:          "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print()
    print("Per-class metrics:")

    for species in SPECIES:

        p = metrics["per_class"][species]["precision"]
        r = metrics["per_class"][species]["recall"]

        print(
            f"  {species:<10} "
            f"precision={p * 100:6.2f}% "
            f"recall={r * 100:6.2f}%"
        )

    print()
    print("Confusion matrix:")
    print(
        f"{'Actual':<15}"
        f"{'setosa':>10}"
        f"{'versicolor':>14}"
        f"{'virginica':>12}"
    )

    for actual in SPECIES:

        print(
            f"{actual:<15}"
            f"{metrics['confusion_matrix'][actual]['setosa']:>10}"
            f"{metrics['confusion_matrix'][actual]['versicolor']:>14}"
            f"{metrics['confusion_matrix'][actual]['virginica']:>12}"
        )

    print()
    print(f"Saved: {output_path}")

    return output_data


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("WEEK 10 - GEMINI FINE-TUNING EVALUATION")
    print("=" * 70)

    print()
    print("Project:", PROJECT_ID)
    print("Location:", LOCATION)

    print()
    print("V1 endpoint:", V1_ENDPOINT)
    print("V2 endpoint:", V2_ENDPOINT)

    token = get_access_token()

    # --------------------------------------------------------
    # V1
    # --------------------------------------------------------

    v1 = evaluate_model(
        name="V1 Raw",
        endpoint=V1_ENDPOINT,
        test_file=V1_TEST_FILE,
        format_checker=check_format_v1,
        token=token,
    )

    # --------------------------------------------------------
    # V2
    # --------------------------------------------------------

    v2 = evaluate_model(
        name="V2 Description",
        endpoint=V2_ENDPOINT,
        test_file=V2_TEST_FILE,
        format_checker=check_format_v2,
        token=token,
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    m1 = v1["metrics"]
    m2 = v2["metrics"]

    if m1["accuracy"] > m2["accuracy"]:
        winner = "V1 Raw"

    elif m2["accuracy"] > m1["accuracy"]:
        winner = "V2 Description"

    else:
        winner = "Tie"

    comparison = {
        "v1": {
            "endpoint": V1_ENDPOINT,
            "accuracy": m1["accuracy"],
            "format_compliance": m1["format_compliance"],
            "macro_precision": m1["macro_precision"],
            "macro_recall": m1["macro_recall"],
        },
        "v2": {
            "endpoint": V2_ENDPOINT,
            "accuracy": m2["accuracy"],
            "format_compliance": m2["format_compliance"],
            "macro_precision": m2["macro_precision"],
            "macro_recall": m2["macro_recall"],
        },
        "winner_by_accuracy": winner,
        "accuracy_difference_v2_minus_v1": (
            m2["accuracy"] - m1["accuracy"]
        ),
        "format_compliance_difference_v2_minus_v1": (
            m2["format_compliance"]
            - m1["format_compliance"]
        ),
    }

    comparison_path = os.path.join(
        RESULTS_DIR,
        "v1_vs_v2_comparison.json",
    )

    with open(comparison_path, "w") as f:
        json.dump(
            comparison,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V1 VS V2 COMPARISON")
    print("=" * 70)

    print(
        f"V1 Accuracy:          "
        f"{m1['accuracy'] * 100:.2f}%"
    )

    print(
        f"V2 Accuracy:          "
        f"{m2['accuracy'] * 100:.2f}%"
    )

    print()

    print(
        f"V1 Format Compliance: "
        f"{m1['format_compliance'] * 100:.2f}%"
    )

    print(
        f"V2 Format Compliance: "
        f"{m2['format_compliance'] * 100:.2f}%"
    )

    print()

    print(
        f"V1 Macro Precision:    "
        f"{m1['macro_precision'] * 100:.2f}%"
    )

    print(
        f"V2 Macro Precision:    "
        f"{m2['macro_precision'] * 100:.2f}%"
    )

    print()

    print(
        f"V1 Macro Recall:       "
        f"{m1['macro_recall'] * 100:.2f}%"
    )

    print(
        f"V2 Macro Recall:       "
        f"{m2['macro_recall'] * 100:.2f}%"
    )

    print()

    print(
        "Winner by accuracy:",
        winner,
    )

    print()
    print(
        f"Saved comparison: "
        f"{comparison_path}"
    )

    print()
    print("=" * 70)
    print("TASK 4 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
