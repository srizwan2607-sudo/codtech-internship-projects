# PROJECT REPORT
## Image Classifier using Convolutional Neural Network (CNN)

### 1. Objective
Build and evaluate a CNN that classifies handwritten digit images into ten classes (0-9).

### 2. Dataset
The scikit-learn digits dataset is normalized and reshaped to include a grayscale channel.

### 3. CNN Architecture
- 32-filter convolution layer
- Max pooling
- 64-filter convolution layer
- Flatten
- Dense layer with 128 units
- Dropout
- 10-class softmax output

### 4. Evaluation
The project records test accuracy and loss, classification metrics, a confusion matrix, and training/validation curves.

### 5. Deliverables
Metrics, classification report, confusion matrix, training plots, sample predictions and a saved trained Keras model are generated.

### 6. Conclusion
This project demonstrates an end-to-end CNN image-classification workflow from data preparation through training, evaluation, visualization and model saving.
