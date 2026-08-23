"""
CODTECH Data Science Internship
Project 1: Medical Diagnosis AI

Educational machine-learning project demonstrating:
- synthetic symptom dataset
- preprocessing
- train/test split
- comparison of Logistic Regression, Decision Tree and Random Forest
- evaluation metrics
- confusion matrix and feature importance
- example prediction

IMPORTANT: This is an educational prototype, NOT a medical diagnostic system.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)

# Synthetic educational dataset. Symptoms are binary (1=present, 0=absent).
# It is intentionally labeled as synthetic so results are not presented as
# clinically validated medical knowledge.
data = [
[1,1,1,1,1,"Flu"], [1,1,0,1,1,"Flu"], [1,1,1,1,0,"Flu"],
[1,0,0,1,1,"Flu"], [1,1,0,1,0,"Flu"], [1,0,1,1,0,"Flu"],
[0,1,1,0,0,"Common Cold"], [0,1,0,1,0,"Common Cold"],
[0,1,0,0,1,"Common Cold"], [0,1,1,0,1,"Common Cold"],
[0,1,0,1,1,"Common Cold"], [0,0,1,0,0,"Migraine"],
[0,0,1,1,0,"Migraine"], [0,0,1,1,1,"Migraine"],
[0,0,1,0,1,"Migraine"], [0,0,1,1,0,"Migraine"],
[0,0,0,0,0,"Healthy"], [0,0,0,0,0,"Healthy"],
[0,0,0,0,0,"Healthy"], [0,0,0,1,0,"Healthy"],
[0,0,0,0,1,"Healthy"], [0,0,0,1,0,"Healthy"],
]

columns = ["fever","cough","headache","fatigue","body_pain","diagnosis"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT / "synthetic_symptom_dataset.csv", index=False)

X = df.drop(columns="diagnosis")
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=5, random_state=42
    ),
}

results = []
trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    trained[name] = model
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0),
    })

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
results_df.to_csv(OUTPUT / "model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = trained[best_name]
best_pred = best_model.predict(X_test)

with open(OUTPUT / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write(classification_report(y_test, best_pred, zero_division=0))

# Model comparison chart
ax = results_df.set_index("Model")[["Accuracy","Precision","Recall","F1 Score"]].plot(
    kind="bar", ylim=(0,1.05), figsize=(9,5), title="Model Performance Comparison"
)
ax.set_ylabel("Score")
ax.set_xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "model_comparison.png", dpi=180)
plt.close()

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, xticks_rotation=30
)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(OUTPUT / "confusion_matrix.png", dpi=180)
plt.close()

# Feature importance where supported
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(
        best_model.feature_importances_, index=X.columns
    ).sort_values(ascending=True)
    ax = importance.plot(kind="barh", figsize=(8,5), title="Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT / "feature_importance.png", dpi=180)
    plt.close()

# Example prediction
example = pd.DataFrame([{
    "fever": 1, "cough": 1, "headache": 0,
    "fatigue": 1, "body_pain": 1
}])
prediction = best_model.predict(example)[0]

with open(OUTPUT / "example_prediction.txt", "w", encoding="utf-8") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"Example symptom profile: {example.iloc[0].to_dict()}\n")
    f.write(f"Predicted educational category: {prediction}\n")
    f.write("\nThis output is not medical advice or a diagnosis.\n")

print("Project completed successfully.")
print(f"Best model: {best_name}")
print(results_df.to_string(index=False))
print(f"\nExample educational prediction: {prediction}")
print(f"Outputs saved in: {OUTPUT}")
