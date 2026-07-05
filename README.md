# MLOps Weekly Assignment – Week 3

**Name:** Udghosh Rao  
**Roll Number:** 24F2000612

## Overview

This assignment demonstrates how to integrate Feast Feature Store into the IRIS machine learning pipeline. The project uses Feast to manage features for both model training and inference so that the same feature definitions are used in both stages.

## Project Structure

```
.
├── 24F2000612_Assignment_3_May2026_MLOps.ipynb
├── README.md
└── iris_feature_repo
    └── feature_repo
        ├── feature_store.yaml
        ├── iris_features.py
        ├── iris_raw.csv
        ├── data
        ├── registry.db
        ├── online_store.db
        └── iris.parquet
```

## Tasks Completed

- Initialized a Feast feature repository.
- Created an entity for the IRIS dataset.
- Defined a file source and feature view.
- Applied the Feast definitions.
- Materialized features into the online store.
- Retrieved historical features from the offline store for model training.
- Retrieved online features for inference.
- Verified that training and inference use the same feature values.

## Requirements

- Python 3
- Feast
- Pandas
- Scikit-learn
- PyArrow

## Running the Project

1. Open the assignment notebook.
2. Navigate to the Feast repository.
3. Run:

```bash
feast apply
```

4. Materialize features:

```bash
feast materialize-incremental $(date +%Y-%m-%d)
```

5. Run the notebook to train the model and perform inference.

## Notes

This project was developed and tested on Google Cloud Platform (Vertex AI Workbench) as part of the IIT Madras BS Degree MLOps weekly assignment.
