from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings('ignore')
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

def create_dataset(n_samples=3000):
    rng = np.random.default_rng(RANDOM_STATE)
    amount = rng.lognormal(mean=4.2, sigma=0.9, size=n_samples)
    account_age_days = rng.integers(30, 4000, n_samples)
    transactions_last_24h = rng.poisson(4, n_samples) + 1
    international = rng.binomial(1, 0.18, n_samples)
    unusual_device = rng.binomial(1, 0.12, n_samples)
    unusual_location = rng.binomial(1, 0.10, n_samples)
    failed_attempts = rng.poisson(0.35, n_samples)
    distance_from_home_km = rng.gamma(2.0, 18.0, n_samples)
    score = (-4.2 + 0.0009*amount + 0.32*transactions_last_24h + 1.25*international + 1.55*unusual_device + 1.40*unusual_location + 0.75*failed_attempts + 0.025*distance_from_home_km - 0.00018*account_age_days)
    probability = 1 / (1 + np.exp(-score))
    fraud = rng.binomial(1, probability)
    return pd.DataFrame({
        'transaction_amount': amount.round(2), 'account_age_days': account_age_days,
        'transactions_last_24h': transactions_last_24h, 'international_transaction': international,
        'unusual_device': unusual_device, 'unusual_location': unusual_location,
        'failed_login_attempts': failed_attempts, 'distance_from_home_km': distance_from_home_km.round(2),
        'fraud': fraud})

def main():
    print('Creating synthetic banking transaction dataset...')
    df = create_dataset()
    df.to_csv(OUTPUT_DIR / 'synthetic_banking_transactions.csv', index=False)
    X, y = df.drop(columns=['fraud']), df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    models = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    }
    results, predictions = [], {}
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        results.append({'Model': name, 'Accuracy': accuracy_score(y_test, pred), 'Precision': precision_score(y_test, pred, zero_division=0), 'Recall': recall_score(y_test, pred, zero_division=0), 'F1 Score': f1_score(y_test, pred, zero_division=0)})
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    ax = results_df.set_index('Model')[['Accuracy','Precision','Recall','F1 Score']].plot(kind='bar', figsize=(10,6), ylim=(0,1.05))
    ax.set_title('Fraud Detection Model Comparison'); ax.set_ylabel('Score'); ax.set_xlabel('Model'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150); plt.close()
    best_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']; best_pred = predictions[best_name]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, best_pred), display_labels=['Legitimate','Fraud']); disp.plot(); plt.title(f'Confusion Matrix - {best_name}'); plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150); plt.close()
    with open(OUTPUT_DIR / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(f'Best model: {best_name}\n\n'); f.write(classification_report(y_test, best_pred, target_names=['Legitimate','Fraud'], zero_division=0))
    if best_name in {'Decision Tree','Random Forest'}:
        fitted = models[best_name]
        feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': fitted.feature_importances_}).sort_values('Importance', ascending=False)
        feature_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
        ax = feature_df.sort_values('Importance').plot(x='Feature', y='Importance', kind='barh', figsize=(9,6), legend=False); ax.set_title(f'Feature Importance - {best_name}'); ax.set_xlabel('Importance'); plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150); plt.close()
    example = pd.DataFrame([{'transaction_amount':1850.0,'account_age_days':120,'transactions_last_24h':12,'international_transaction':1,'unusual_device':1,'unusual_location':1,'failed_login_attempts':2,'distance_from_home_km':220.0}])
    label = 'Fraud' if int(models[best_name].predict(example)[0]) else 'Legitimate'
    with open(OUTPUT_DIR / 'example_prediction.txt', 'w', encoding='utf-8') as f:
        f.write(f'Best model: {best_name}\nExample transaction prediction: {label}\n\nTransaction details:\n{example.to_string(index=False)}')
    print('\nProject completed successfully.')
    print(f'Best model: {best_name}')
    print(results_df.to_string(index=False))
    print(f'\nOutputs saved in: {OUTPUT_DIR}')

if __name__ == '__main__': main()
