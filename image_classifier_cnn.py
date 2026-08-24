"""
CODTECH Internship - Task 3
Image Classifier using Convolutional Neural Network (CNN)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        print("TensorFlow is not installed. Run: python -m pip install -r requirements.txt")
        return

    np.random.seed(42)
    tf.random.set_seed(42)

    digits = load_digits()
    X = (digits.images.astype("float32") / 16.0)[..., np.newaxis]
    y = digits.target
    y_cat = to_categorical(y, 10)

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y_cat, y, test_size=0.20, random_state=42, stratify=y
    )

    print("=" * 60)
    print("CODTECH INTERNSHIP - IMAGE CLASSIFIER USING CNN")
    print("=" * 60)
    print(f"Dataset: {len(X)} images")
    print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(8,8,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.30),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("\nTraining CNN...")
    history = model.fit(X_train, y_train, validation_split=0.15,
                         epochs=12, batch_size=32, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test, verbose=0).argmax(axis=1)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    pd.DataFrame([{"Model":"Convolutional Neural Network",
                   "Test Accuracy":accuracy, "Test Loss":loss}]).to_csv(
        OUTPUT_DIR/"cnn_metrics.csv", index=False)

    report = classification_report(y_test_raw, predictions,
                                    target_names=[str(i) for i in range(10)],
                                    zero_division=0)
    (OUTPUT_DIR/"classification_report.txt").write_text(
        "CNN Image Classifier - Classification Report\n"+"="*50+"\n"+report,
        encoding="utf-8")

    cm = confusion_matrix(y_test_raw, predictions)
    plt.figure(figsize=(8,6))
    plt.imshow(cm)
    plt.title("CNN Confusion Matrix - Digits")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.colorbar()
    plt.xticks(range(10)); plt.yticks(range(10))
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"confusion_matrix.png", dpi=150); plt.close()

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(OUTPUT_DIR/"training_history.csv", index=False)

    for key, title, filename, ylabel in [
        ("accuracy","CNN Training and Validation Accuracy","training_accuracy.png","Accuracy"),
        ("loss","CNN Training and Validation Loss","training_loss.png","Loss")]:
        plt.figure(figsize=(9,5))
        plt.plot(history.history[key], label="Training "+ylabel)
        plt.plot(history.history["val_"+key], label="Validation "+ylabel)
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend()
        plt.tight_layout(); plt.savefig(OUTPUT_DIR/filename, dpi=150); plt.close()

    n = min(10, len(X_test))
    pd.DataFrame({"Actual":y_test_raw[:n], "Predicted":predictions[:n]}).to_csv(
        OUTPUT_DIR/"sample_predictions.csv", index=False)

    model.save(OUTPUT_DIR/"cnn_digit_classifier.keras")
    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print(f"Outputs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
