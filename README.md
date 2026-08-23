# CODTECH Medical Diagnosis AI

## Project 1 — Data Science Internship

### Project objective
Build an educational machine-learning prototype that classifies a **synthetic symptom dataset** into broad categories. The project demonstrates a complete data-science workflow rather than relying on a single model.

### Models compared
1. Logistic Regression
2. Decision Tree
3. Random Forest

### Workflow
**Data → preprocessing → train/test split → model training → evaluation → model comparison → visualization → example prediction**

### Dataset
The included dataset is **synthetic and intentionally small**. It contains five binary symptom features:
- fever
- cough
- headache
- fatigue
- body pain

Target categories:
- Flu
- Common Cold
- Migraine
- Healthy

The dataset is not sourced from patients and is not clinically validated.

### Evaluation
The script calculates:
- Accuracy
- Precision
- Recall
- F1 score
- Classification report
- Confusion matrix

It also generates model-comparison and feature-importance visualizations.

### Run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Then:

```bash
python medical_diagnosis_ai.py
```

Generated files appear in the `outputs` folder.

### Project structure

```text
CODTECH_Medical_Diagnosis_AI_A1/
│
├── medical_diagnosis_ai.py
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
└── outputs/
    ├── synthetic_symptom_dataset.csv
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── classification_report.txt
    └── example_prediction.txt
```

### Important disclaimer
This is an academic demonstration only. It must **not** be used to diagnose, treat, or make healthcare decisions for real people.
