# 24f2000612_MLOPS_WEEKLY_ASSIGNMENT — Week 4

## Continuous Integration for the IRIS Pipeline

This branch (`week_4`) adds CI to the IRIS pipeline using GitHub Actions,
so every push and pull request automatically fetches the versioned data and
model via DVC, runs a pytest suite, and posts a results report as a comment
on the PR via CML.

## What Was Built

| File | Purpose |
|---|---|
| `tests/test_data_validation.py` | Validates `data/iris.csv`: schema, missing values, numeric types, plausible value ranges, valid species labels, and duplicate-row tolerance. |
| `tests/test_model_evaluation.py` | Loads `model.joblib`, rebuilds the same eval split used in `train.py`, and asserts minimum accuracy (0.85) and macro F1 (0.80). |
| `.github/workflows/ci.yml` | GitHub Actions workflow: checks out the repo, installs dependencies, authenticates to GCS, runs `dvc pull`, runs the pytest suite, and posts a CML comment on pull requests. |
| `requirements.txt` | Pinned dependencies for the CI environment, including `scikit-learn==1.3.2` to match the version the committed model was trained with. |

## Step-by-Step Summary of What Was Done

1. **Task 1 — Data validation tests**: Wrote `tests/test_data_validation.py`
   covering row count, expected columns, missing values, numeric dtypes,
   plausible value ranges (0-15 cm), valid species labels, and per-species
   minimum row counts.
2. **Task 2 — Model evaluation tests**: Wrote `tests/test_model_evaluation.py`,
   which loads the trained model and evaluates it on a held-out split,
   asserting minimum accuracy and macro F1 thresholds.
3. **Task 3 — GitHub Actions + DVC**: Built `.github/workflows/ci.yml` to
   checkout the repo, install `requirements.txt`, authenticate to GCS using
   a service account key stored as the `GCP_SA_KEY` GitHub secret, run
   `dvc pull` to fetch `data/iris.csv` and `model.joblib`, then run the
   pytest suite.
4. **Task 4 — Trigger on every push/PR**: The workflow's `on:` block has no
   branch filter (`push:` / `pull_request:` with no `branches:` restriction),
   so CI runs on every branch, not just `main`.
5. **Task 5 — CML report on PR**: Added a CML step (`iterative/setup-cml`)
   that only runs on `pull_request` events, formats the pytest output into
   a markdown report, and posts it as a PR comment via `cml comment create`.
6. **Task 6 — Merge via PR**: Opened a pull request from `week_4` into
   `main`, confirmed CI ran automatically and the CML comment appeared with
   the test report, then merged.

## Errors Encountered and How They Were Resolved

- **Missing `.dvc/config` in git history**: The DVC remote configuration
  had never been committed to this repo, so `dvc pull` failed in CI with
  `not inside of a DVC repository`. Resolved by recovering the config from
  an earlier commit (`git show <commit>:.dvc/config`) and re-adding it,
  along with the missing `data/iris.csv.dvc` pointer file.
- **scikit-learn version mismatch**: The committed `model.joblib` was
  trained with scikit-learn 1.3.2, but CI installed the latest version by
  default, causing `AttributeError: 'DecisionTreeClassifier' object has no
  attribute 'monotonic_cst'` on load. Resolved by pinning
  `scikit-learn==1.3.2` in `requirements.txt`.
- **Minor duplicate rows from augmentation**: `augment_data.py` resamples
  rows with noise and a fixed random seed, which can produce a small number
  of exact duplicate rows. The data validation test was adjusted to allow
  up to 5% duplication (a real, expected side effect of augmentation)
  rather than requiring zero duplicates.

## Cloud / Local Resources Used

- **Vertex AI Workbench** — used to run all commands and edit files.
- **GitHub Actions** — hosted CI runner (`ubuntu-latest`).
- **Google Cloud Storage** — DVC remote (`gs://24f2000612-mlops-week2/dvc-store`).
- **CML (iterative/setup-cml)** — posts test reports as PR comments.

## Author

Udghosh Rao — 24f2000612, B.S. Data Science & Applications, IIT Madras.
