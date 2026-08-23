import json
import time
import subprocess
import urllib.request
import urllib.error

PROJECT_ID = "project-83abca74-808e-4448-be0"
LOCATION = "us-central1"

JOB_ID = "8025104619724079104"

JOB_NAME = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/tuningJobs/{JOB_ID}"
)

API_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/tuningJobs/{JOB_ID}"
)

print("=" * 70)
print("IRIS WEEK 10 - GEMINI V1 RAW FINE-TUNING")
print("=" * 70)
print(f"Project      : {PROJECT_ID}")
print(f"Location     : {LOCATION}")
print(f"Job ID       : {JOB_ID}")
print(f"Job resource : {JOB_NAME}")
print("=" * 70)


def get_access_token():
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_job():
    token = get_access_token()

    request = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print("HTTP ERROR:", e.code)
        print(body)
        raise


print()
print("Checking existing tuning job...")
print()

job = get_job()

state = job.get("state", "UNKNOWN")

print("Current state:", state)

if job.get("error"):
    print()
    print("JOB ERROR:")
    print(json.dumps(job["error"], indent=2))


terminal_states = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


while state not in terminal_states:

    print()
    print("-" * 70)
    print("Job still running.")
    print("State:", state)
    print("Waiting 30 seconds...")
    print("-" * 70)

    time.sleep(30)

    job = get_job()
    state = job.get("state", "UNKNOWN")

    print()
    print("Updated state:", state)


print()
print("=" * 70)
print("TUNING JOB FINISHED")
print("=" * 70)

print("State:", state)

print()
print("Full job response:")
print(json.dumps(job, indent=2))


# ------------------------------------------------------------
# Save complete tuning-job response
# ------------------------------------------------------------

output_file = "week10_v1_tuning_job.json"

with open(output_file, "w") as f:
    json.dump(job, f, indent=2)

print()
print("Saved job information to:")
print(output_file)


# ------------------------------------------------------------
# Check success
# ------------------------------------------------------------

if state != "JOB_STATE_SUCCEEDED":

    print()
    print("=" * 70)
    print("TUNING DID NOT SUCCEED")
    print("=" * 70)

    if job.get("error"):
        print(json.dumps(job["error"], indent=2))

    raise SystemExit(1)


# ------------------------------------------------------------
# Extract tuned model information
# ------------------------------------------------------------

tuned_model = job.get("tunedModel", {})

print()
print("=" * 70)
print("TUNED MODEL")
print("=" * 70)

if isinstance(tuned_model, dict):

    model_name = tuned_model.get("model")
    endpoint = tuned_model.get("endpoint")
    checkpoints = tuned_model.get("checkpoints")

    print("Model:")
    print(model_name)

    print()
    print("Endpoint:")
    print(endpoint)

    if checkpoints:
        print()
        print("Checkpoints:")

        for i, checkpoint in enumerate(checkpoints, start=1):
            print()
            print(f"Checkpoint {i}:")
            print(json.dumps(checkpoint, indent=2))

else:
    print(tuned_model)


# ------------------------------------------------------------
# Save important outputs
# ------------------------------------------------------------

result = {
    "project_id": PROJECT_ID,
    "location": LOCATION,
    "job_id": JOB_ID,
    "job_name": JOB_NAME,
    "state": state,
    "tuned_model": tuned_model,
    "base_model": job.get("baseModel"),
    "training_dataset": (
        job.get("supervisedTuningSpec", {})
        .get("trainingDatasetUri")
    ),
    "validation_dataset": (
        job.get("supervisedTuningSpec", {})
        .get("validationDatasetUri")
    ),
    "hyper_parameters": (
        job.get("supervisedTuningSpec", {})
        .get("hyperParameters")
    ),
}

result_file = "week10_v1_result.json"

with open(result_file, "w") as f:
    json.dump(result, f, indent=2)

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(json.dumps(result, indent=2))

print()
print(f"Saved final result to: {result_file}")
print()
print("WEEK 10 V1 TUNING COMPLETE")
print("=" * 70)
