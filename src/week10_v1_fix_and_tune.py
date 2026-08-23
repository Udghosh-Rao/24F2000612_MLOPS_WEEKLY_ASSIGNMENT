import json
import subprocess
import time
import urllib.request
import urllib.error

PROJECT_ID = "project-83abca74-808e-4448-be0"
LOCATION = "us-central1"
BUCKET = "24f2000612-mlops-oppe1-dvc-20260802"

RAW_TRAIN_URI = (
    f"gs://{BUCKET}/week_10/training/iris_v1_raw.jsonl"
)

RAW_VALIDATION_URI = (
    f"gs://{BUCKET}/week_10/validation/iris_v1_raw.jsonl"
)

FIXED_TRAIN_LOCAL = "iris_v1_generatecontent_train.jsonl"
FIXED_VALIDATION_LOCAL = "iris_v1_generatecontent_validation.jsonl"

FIXED_TRAIN_URI = (
    f"gs://{BUCKET}/week_10/training/"
    f"iris_v1_generatecontent.jsonl"
)

FIXED_VALIDATION_URI = (
    f"gs://{BUCKET}/week_10/validation/"
    f"iris_v1_generatecontent.jsonl"
)


def gcloud_storage_cat(uri):
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def convert_dataset(raw_uri, output_file):
    print()
    print("=" * 70)
    print("CONVERTING DATASET")
    print("=" * 70)
    print("SOURCE :", raw_uri)
    print("OUTPUT :", output_file)

    raw = gcloud_storage_cat(raw_uri)

    input_lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip()
    ]

    converted = []

    for line_number, line in enumerate(input_lines, start=1):

        row = json.loads(line)

        if "input_text" not in row:
            raise ValueError(
                f"Line {line_number}: missing input_text"
            )

        if "output_text" not in row:
            raise ValueError(
                f"Line {line_number}: missing output_text"
            )

        input_text = str(row["input_text"])
        output_text = str(row["output_text"])

        example = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": input_text
                        }
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": output_text
                        }
                    ]
                }
            ]
        }

        converted.append(example)

    with open(output_file, "w", encoding="utf-8") as f:
        for example in converted:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("Examples converted:", len(converted))

    return converted


def validate_jsonl(path, expected_count):
    print()
    print("=" * 70)
    print("VALIDATING")
    print("=" * 70)
    print("FILE:", path)

    with open(path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if len(lines) != expected_count:
        raise ValueError(
            f"Expected {expected_count} examples, "
            f"found {len(lines)}"
        )

    for line_number, line in enumerate(lines, start=1):

        obj = json.loads(line)

        if "contents" not in obj:
            raise ValueError(
                f"Line {line_number}: missing contents"
            )

        contents = obj["contents"]

        if len(contents) != 2:
            raise ValueError(
                f"Line {line_number}: expected 2 contents messages"
            )

        if contents[0].get("role") != "user":
            raise ValueError(
                f"Line {line_number}: first role is not user"
            )

        if contents[1].get("role") != "model":
            raise ValueError(
                f"Line {line_number}: second role is not model"
            )

        if not contents[0]["parts"][0].get("text"):
            raise ValueError(
                f"Line {line_number}: empty user text"
            )

        if not contents[1]["parts"][0].get("text"):
            raise ValueError(
                f"Line {line_number}: empty model text"
            )

    print("Validation: PASSED")
    print("Examples:", len(lines))

    print()
    print("First converted example:")
    print(json.dumps(
        json.loads(lines[0]),
        indent=2
    ))


def upload(local_file, remote_uri):
    print()
    print("=" * 70)
    print("UPLOADING")
    print("=" * 70)
    print("LOCAL :", local_file)
    print("REMOTE:", remote_uri)

    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            local_file,
            remote_uri,
        ],
        check=True,
    )

    print("Upload: SUCCESS")


def get_token():
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def create_tuning_job():

    print()
    print("=" * 70)
    print("CREATING CORRECTED GEMINI TUNING JOB")
    print("=" * 70)

    token = get_token()

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/tuningJobs"
    )

    payload = {
        "baseModel": "gemini-2.5-flash",

        "supervisedTuningSpec": {
            "trainingDatasetUri": FIXED_TRAIN_URI,
            "validationDatasetUri": FIXED_VALIDATION_URI,

            "hyperParameters": {
                "epochCount": 3,
                "adapterSize": 4,
                "learningRateMultiplier": 1.0
            }
        },

        "tunedModelDisplayName": "iris-gemini-v1-generatecontent"
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")

        print()
        print("TUNING JOB CREATION FAILED")
        print("HTTP:", e.code)
        print(body)

        raise

    print()
    print("TUNING JOB CREATED")

    print(json.dumps(result, indent=2))

    return result


def get_job(job_name):

    token = get_token()

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"{job_name}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )

    with urllib.request.urlopen(request) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def wait_for_job(job_name):

    print()
    print("=" * 70)
    print("WAITING FOR TUNING JOB")
    print("=" * 70)

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:

        job = get_job(job_name)

        state = job.get("state", "UNKNOWN")

        print(
            time.strftime("%H:%M:%S"),
            "STATE:",
            state
        )

        if state in terminal_states:
            return job

        time.sleep(30)


def main():

    print("=" * 70)
    print("WEEK 10 - V1 GEMINI DATA FORMAT FIX")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Convert training data
    # ---------------------------------------------------------

    train_examples = convert_dataset(
        RAW_TRAIN_URI,
        FIXED_TRAIN_LOCAL
    )

    # ---------------------------------------------------------
    # 2. Convert validation data
    # ---------------------------------------------------------

    validation_examples = convert_dataset(
        RAW_VALIDATION_URI,
        FIXED_VALIDATION_LOCAL
    )

    # ---------------------------------------------------------
    # 3. Validate
    # ---------------------------------------------------------

    validate_jsonl(
        FIXED_TRAIN_LOCAL,
        len(train_examples)
    )

    validate_jsonl(
        FIXED_VALIDATION_LOCAL,
        len(validation_examples)
    )

    # ---------------------------------------------------------
    # 4. Upload
    # ---------------------------------------------------------

    upload(
        FIXED_TRAIN_LOCAL,
        FIXED_TRAIN_URI
    )

    upload(
        FIXED_VALIDATION_LOCAL,
        FIXED_VALIDATION_URI
    )

    # ---------------------------------------------------------
    # 5. Verify uploaded files
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("VERIFYING GCS FILES")
    print("=" * 70)

    subprocess.run(
        ["gcloud", "storage", "ls", FIXED_TRAIN_URI],
        check=True
    )

    subprocess.run(
        ["gcloud", "storage", "ls", FIXED_VALIDATION_URI],
        check=True
    )

    # ---------------------------------------------------------
    # 6. Create new tuning job
    # ---------------------------------------------------------

    created_job = create_tuning_job()

    job_name = created_job["name"]

    print()
    print("NEW JOB:")
    print(job_name)

    # ---------------------------------------------------------
    # 7. Save initial job
    # ---------------------------------------------------------

    with open(
        "week10_v1_generatecontent_job_created.json",
        "w"
    ) as f:
        json.dump(created_job, f, indent=2)

    # ---------------------------------------------------------
    # 8. Wait
    # ---------------------------------------------------------

    final_job = wait_for_job(job_name)

    # ---------------------------------------------------------
    # 9. Save final job
    # ---------------------------------------------------------

    with open(
        "week10_v1_generatecontent_job_final.json",
        "w"
    ) as f:
        json.dump(final_job, f, indent=2)

    # ---------------------------------------------------------
    # 10. Print final result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TUNING RESULT")
    print("=" * 70)

    print(json.dumps(final_job, indent=2))

    state = final_job.get("state")

    if state != "JOB_STATE_SUCCEEDED":

        print()
        print("=" * 70)
        print("TUNING FAILED")
        print("=" * 70)

        if final_job.get("error"):
            print(
                json.dumps(
                    final_job["error"],
                    indent=2
                )
            )

        raise SystemExit(1)

    print()
    print("=" * 70)
    print("TUNING SUCCEEDED")
    print("=" * 70)

    tuned_model = final_job.get("tunedModel")

    print()
    print("TUNED MODEL:")
    print(json.dumps(tuned_model, indent=2))

    result = {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "job_name": job_name,
        "state": state,
        "base_model": final_job.get("baseModel"),
        "training_dataset": FIXED_TRAIN_URI,
        "validation_dataset": FIXED_VALIDATION_URI,
        "hyper_parameters": (
            final_job
            .get("supervisedTuningSpec", {})
            .get("hyperParameters")
        ),
        "tuned_model": tuned_model,
    }

    with open(
        "week10_v1_final_result.json",
        "w"
    ) as f:
        json.dump(result, f, indent=2)

    print()
    print("Saved:")
    print("  week10_v1_generatecontent_job_created.json")
    print("  week10_v1_generatecontent_job_final.json")
    print("  week10_v1_final_result.json")

    print()
    print("=" * 70)
    print("WEEK 10 V1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
