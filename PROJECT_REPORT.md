# CODTECH Internship — Project 1
# Medical Diagnosis AI

## 1. Abstract
This project presents an educational machine-learning pipeline for classifying a synthetic symptom dataset. Three classification algorithms are compared to demonstrate how a data-science project can move from structured data to model evaluation and interpretation.

## 2. Problem Statement
Manual rule-based classification can become difficult when multiple features interact. The objective here is to demonstrate how supervised machine learning can learn patterns from labeled examples and classify new symptom profiles.

## 3. Objectives
- Create a reproducible synthetic dataset.
- Prepare features and labels.
- Split the dataset into training and testing sets.
- Train multiple classification algorithms.
- Compare their performance using standard metrics.
- Generate visual evaluation outputs.
- Demonstrate a prediction on an example profile.

## 4. Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib

## 5. Methodology
1. Define binary symptom features.
2. Store the synthetic records in a Pandas DataFrame.
3. Separate features (`X`) and target (`y`).
4. Perform a stratified train/test split.
5. Train Logistic Regression, Decision Tree and Random Forest models.
6. Calculate accuracy, precision, recall and F1 score.
7. Select the model with the highest test F1 score.
8. Produce a confusion matrix and model-comparison chart.
9. Save all outputs for reproducibility.

## 6. Results
The exact scores are generated when the script runs and are saved in:
`outputs/model_comparison.csv`

The classification report is saved in:
`outputs/classification_report.txt`

This keeps the reported results tied directly to the reproducible code rather than manually entering numbers.

## 7. Conclusion
The project demonstrates a complete introductory supervised-learning workflow and shows why comparing several algorithms is more informative than presenting one model alone.

## 8. Limitations and Ethics
The dataset is synthetic, very small, and not clinically validated. Real medical systems require large representative datasets, rigorous validation, clinical oversight, privacy protections, bias assessment, and regulatory compliance. Therefore, this project is strictly an educational prototype and not a diagnostic system.

## 9. Future Improvements
- Use a properly licensed, clinically validated dataset.
- Add robust cross-validation and hyperparameter tuning.
- Add missing-value and class-imbalance handling.
- Build an interactive Streamlit interface.
- Add explainability methods such as SHAP.
- Conduct fairness and robustness testing.
