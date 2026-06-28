# IITM BS MLOps - Week 2 Assignment

## Integrating DVC into the IRIS Machine Learning Pipeline

**Student Name:** Udghosh Rao
**IITM BS Roll Number:** 24F2000612
**Course:** MLOps
**Week:** 2
**Branch:** `week_2`

---

# Objective

This assignment extends the Week 1 IRIS Machine Learning pipeline by integrating **Data Version Control (DVC)**. The objective is to build a reproducible machine learning workflow where datasets and trained models are version-controlled using Git and DVC while storing large artifacts in Google Cloud Storage (GCS).

---

# Assignment Objectives

* Initialize DVC in the Git repository.
* Configure Google Cloud Storage as the DVC remote.
* Version multiple iterations of the IRIS dataset.
* Track trained model artifacts using DVC.
* Demonstrate switching between historical versions using Git and DVC checkout.
* Push all project files to GitHub while storing large files in GCS.

---

# Project Structure

```text
24F2000612_MLOPS_WEEKLY_ASSIGNMENT/
│
├── data/
│   ├── iris.csv
│   ├── iris.csv.dvc
│
├── models/
│   ├── model.pkl
│   ├── model.pkl.dvc
│
├── scripts/
│   ├── train.py
│   ├── inference.py
│
├── .dvc/
├── .dvcignore
├── .gitignore
├── dvc.yaml (optional)
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python 3.x
* Scikit-learn
* Pandas
* NumPy
* Git
* DVC
* Google Cloud Storage
* Google Cloud Platform
* Vertex AI Workbench

---

# DVC Remote Configuration

The project uses Google Cloud Storage as the remote backend for DVC.

Example configuration:

```bash
dvc remote add -d gcsremote gs://<your-bucket-name>
```

Verify configuration:

```bash
dvc remote list
```

---

# Workflow

## Step 1 — Initialize DVC

```bash
dvc init
git add .
git commit -m "Initialize DVC"
```

---

## Step 2 — Configure GCS Remote

```bash
dvc remote add -d gcsremote gs://<bucket-name>
```

---

## Step 3 — Track Dataset

```bash
dvc add data/iris.csv
```

---

## Step 4 — Train Model

```bash
python scripts/train.py
```

---

## Step 5 — Track Model

```bash
dvc add models/model.pkl
```

---

## Step 6 — Push Data to Remote Storage

```bash
dvc push
```

---

## Step 7 — Commit Changes

```bash
git add .
git commit -m "Version 2 Dataset and Model"
git push
```

---

# Switching Between Versions

View commit history:

```bash
git log --oneline
```

Checkout previous version:

```bash
git checkout <commit_hash>
dvc checkout
```

Return to latest version:

```bash
git checkout week_2
dvc checkout
```

---

# Files Included

| File          | Description                                |
| ------------- | ------------------------------------------ |
| train.py      | Trains the IRIS classifier                 |
| inference.py  | Loads trained model and performs inference |
| iris.csv      | Training dataset                           |
| iris.csv.dvc  | DVC pointer for dataset                    |
| model.pkl.dvc | DVC pointer for trained model              |
| .dvc          | DVC configuration                          |
| .gitignore    | Git ignore configuration                   |
| README.md     | Project documentation                      |

---

# Reproducibility

To reproduce the project:

Clone the repository.

```bash
git clone <repository-url>
```

Enter the project.

```bash
cd 24F2000612_MLOPS_WEEKLY_ASSIGNMENT
```

Pull DVC files.

```bash
dvc pull
```

Run training.

```bash
python scripts/train.py
```

Run inference.

```bash
python scripts/inference.py
```

---

# Learning Outcomes

Through this assignment, I learned:

* How DVC extends Git for machine learning workflows.
* Configuring Google Cloud Storage as a remote backend.
* Versioning datasets and trained models.
* Restoring historical data and model versions using Git and DVC.
* Building a reproducible and version-controlled machine learning pipeline.

---

# Repository

Repository Name

```text
24F2000612_MLOPS_WEEKLY_ASSIGNMENT
```

Branch

```text
week_2
```

---

# Author

**Udghosh Rao**

IIT Madras BS Degree in Data Science and Applications

Roll Number: **24F2000612**
