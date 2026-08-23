# Week 10 - LLMOps

Fine-tuning Gemini 2.5 Flash on the IRIS dataset using Vertex AI.

## V1 Raw

IRIS feature values are given directly as text.

Example:

`sepal_length: 5.1, sepal_width: 2.5, petal_length: 3.0, petal_width: 1.1`

Output:

`versicolor`

## V2 Description

The same IRIS features are written as a natural-language description.

## Fine-Tuning

Model: Gemini 2.5 Flash

Epochs: 3

Learning rate multiplier: 1.0

V1 tuning job: 4419973118013997056

V2 tuning job: 748694996776910848

Both tuning jobs succeeded.

## Evaluation

Test samples: 23

| Metric | V1 Raw | V2 Description |
|---|---:|---:|
| Accuracy | 86.96% | 34.78% |
| Format Compliance | 0.00% | 0.00% |
| Macro Precision | 90.00% | 43.94% |
| Macro Recall | 87.50% | 37.50% |

## Conclusion

V1 Raw performed better than V2 Description.

V1 achieved 86.96% accuracy compared with 34.78% for V2.

## Tools

- Python
- Google Cloud Storage
- Vertex AI
- Gemini 2.5 Flash
- GitHub
